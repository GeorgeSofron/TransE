"""
10-Fold Cross-Validation for TriModel Evaluation
=================================================
Performs stratified k-fold cross-validation on the DrugBank knowledge graph
to provide robust performance estimates with confidence intervals.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import time

from model import TriModel


# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    "data_path": "drugbank_facts.txt",
    "output_dir": "outputs_trimodel_cv",
    "n_folds": 10,
    "seed": 42,
    
    # Model hyperparameters
    "embedding_dim": 100,
    "learning_rate": 1e-3,
    "batch_size": 1024,
    "epochs": 100,
    "margin": 1.0,
    
    # Early stopping
    "early_stopping_patience": 10,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    
    # Evaluation
    "neg_ratio": 1,  # negatives per positive for evaluation
    
    "device": "cpu",
}


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data Loading
# ----------------------------
def load_triples(path: str) -> pd.DataFrame:
    """Load triples from file."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )
    return df


def make_mappings(df: pd.DataFrame):
    """Create entity and relation mappings from dataframe."""
    entities = pd.Index(pd.concat([df["source"], df["target"]], ignore_index=True).unique())
    relations = pd.Index(df["relation"].unique())
    
    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    return entity2id, relation2id


def encode_triples(df: pd.DataFrame, entity2id: dict, relation2id: dict) -> torch.Tensor:
    """Encode triples as tensor of IDs."""
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


def filter_triples(df: pd.DataFrame, entity2id: dict, relation2id: dict) -> pd.DataFrame:
    """Filter triples to only include known entities and relations."""
    return df[
        (df["source"].isin(entity2id.keys())) &
        (df["relation"].isin(relation2id.keys())) &
        (df["target"].isin(entity2id.keys()))
    ].reset_index(drop=True)


# ----------------------------
# Negative Sampling
# ----------------------------
def build_true_triples_set(triples: torch.Tensor) -> set:
    """Convert tensor of triples to set for O(1) lookup."""
    return set(map(tuple, triples.cpu().numpy().tolist()))


def corrupt_triples(
    pos_triples: torch.Tensor,
    num_entities: int,
    true_triples: set = None,
    max_attempts: int = 10
) -> torch.Tensor:
    """Create negative triples by corrupting head or tail."""
    neg = pos_triples.clone().cpu()
    batch_size = neg.size(0)
    mask = torch.rand(batch_size) < 0.5
    
    for i in range(batch_size):
        h, r, t = neg[i].tolist()
        
        for _ in range(max_attempts):
            random_entity = torch.randint(0, num_entities, (1,)).item()
            
            if mask[i]:
                candidate = (random_entity, r, t)
            else:
                candidate = (h, r, random_entity)
            
            if true_triples is None or candidate not in true_triples:
                if mask[i]:
                    neg[i, 0] = random_entity
                else:
                    neg[i, 2] = random_entity
                break
    
    return neg


def generate_negatives(pos_triples: np.ndarray, num_entities: int,
                       true_triples_set: set, ratio: int = 1) -> np.ndarray:
    """Generate negative triples for evaluation."""
    neg_list = []
    
    for h, r, t in pos_triples:
        for _ in range(ratio):
            for _ in range(10):
                if random.random() < 0.5:
                    new_h = random.randint(0, num_entities - 1)
                    if (new_h, r, t) not in true_triples_set:
                        neg_list.append([new_h, r, t])
                        break
                else:
                    new_t = random.randint(0, num_entities - 1)
                    if (h, r, new_t) not in true_triples_set:
                        neg_list.append([h, r, new_t])
                        break
    
    return np.array(neg_list)


# ----------------------------
# Training
# ----------------------------
@torch.no_grad()
def compute_validation_loss(
    model: TriModel,
    valid_triples: torch.Tensor,
    num_entities: int,
    true_triples: set,
    device: str
) -> float:
    """Compute validation loss."""
    model.eval()
    valid_triples = valid_triples.to(device)
    neg = corrupt_triples(valid_triples, num_entities, true_triples).to(device)
    
    pos_scores = model(valid_triples)
    neg_scores = model(neg)
    
    # Binary cross-entropy style loss for TriModel
    pos_loss = F.softplus(-pos_scores).mean()
    neg_loss = F.softplus(neg_scores).mean()
    
    return (pos_loss + neg_loss).item()


def train_fold(
    train_triples: torch.Tensor,
    valid_triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    config: dict,
    fold: int
) -> TriModel:
    """Train TriModel for one fold."""
    device = config["device"]
    
    model = TriModel(num_entities, num_relations, dim=config["embedding_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config["lr_scheduler_patience"],
        factor=config["lr_scheduler_factor"]
    )
    
    true_triples = build_true_triples_set(train_triples)
    
    best_valid_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_triples = train_triples.to(device)
    n = train_triples.size(0)
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        num_batches = 0
        
        for start in range(0, n, config["batch_size"]):
            idx = perm[start:start + config["batch_size"]]
            pos = train_triples[idx]
            neg = corrupt_triples(pos, num_entities, true_triples).to(device)
            
            pos_scores = model(pos)
            neg_scores = model(neg)
            
            # Pairwise logistic loss
            pos_loss = F.softplus(-pos_scores).mean()
            neg_loss = F.softplus(neg_scores).mean()
            loss = pos_loss + neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        
        # Validation
        valid_loss = compute_validation_loss(
            model, valid_triples, num_entities, true_triples, device
        )
        
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


# ----------------------------
# Evaluation Metrics
# ----------------------------
@torch.no_grad()
def compute_scores(model: TriModel, triples: np.ndarray, device: str) -> np.ndarray:
    """Compute scores for triples."""
    model.eval()
    triples_tensor = torch.tensor(triples, dtype=torch.long).to(device)
    scores = model(triples_tensor).cpu().numpy()
    return scores


def evaluate_fold(
    model: TriModel,
    test_triples: np.ndarray,
    all_triples: np.ndarray,
    num_entities: int,
    config: dict
) -> dict:
    """Evaluate model on test fold."""
    device = config["device"]
    
    # Build true triples set
    true_triples_set = set(map(tuple, all_triples.tolist()))
    
    # Generate negatives
    neg_triples = generate_negatives(
        test_triples, num_entities, true_triples_set, ratio=config["neg_ratio"]
    )
    
    # Compute scores
    pos_scores = compute_scores(model, test_triples, device)
    neg_scores = compute_scores(model, neg_triples, device)
    
    # Labels
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    # Metrics
    metrics = {
        "AUC-ROC": roc_auc_score(y_true, y_scores),
        "AUC-PR": average_precision_score(y_true, y_scores),
        "pos_score_mean": pos_scores.mean(),
        "neg_score_mean": neg_scores.mean(),
        "score_gap": pos_scores.mean() - neg_scores.mean(),
    }
    
    # P@K
    rank_order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[rank_order]
    
    for k in [10, 50, 100]:
        if k <= len(y_true):
            metrics[f"P@{k}"] = np.sum(y_true_sorted[:k]) / k
    
    return metrics


# ----------------------------
# Cross-Validation
# ----------------------------
def run_cross_validation(config: dict):
    """Run 10-fold cross-validation."""
    set_seed(config["seed"])
    
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    print("=" * 70)
    print(f"10-Fold Cross-Validation for TriModel")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {config['data_path']}")
    df = load_triples(config["data_path"])
    print(f"  Total triples: {len(df):,}")
    
    # Create global mappings from all data
    entity2id, relation2id = make_mappings(df)
    print(f"  Entities: {len(entity2id):,}")
    print(f"  Relations: {len(relation2id)}")
    
    # Encode all triples
    all_triples = encode_triples(df, entity2id, relation2id)
    all_triples_np = all_triples.numpy()
    
    # K-Fold split
    kf = KFold(n_splits=config["n_folds"], shuffle=True, random_state=config["seed"])
    
    fold_results = []
    fold_times = []
    
    print(f"\nStarting {config['n_folds']}-fold cross-validation...")
    print("-" * 70)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_triples_np), 1):
        fold_start = time.time()
        print(f"\n[Fold {fold}/{config['n_folds']}]")
        
        # Split data
        train_triples = torch.tensor(all_triples_np[train_idx], dtype=torch.long)
        test_triples_np = all_triples_np[test_idx]
        
        # Further split train into train/valid (90/10)
        n_train = len(train_triples)
        n_valid = int(n_train * 0.1)
        perm = torch.randperm(n_train)
        
        valid_triples = train_triples[perm[:n_valid]]
        train_triples = train_triples[perm[n_valid:]]
        
        print(f"  Train: {len(train_triples):,}, Valid: {len(valid_triples):,}, Test: {len(test_triples_np):,}")
        
        # Train
        print(f"  Training...")
        model = train_fold(
            train_triples=train_triples,
            valid_triples=valid_triples,
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            config=config,
            fold=fold
        )
        
        # Evaluate
        print(f"  Evaluating...")
        metrics = evaluate_fold(
            model=model,
            test_triples=test_triples_np,
            all_triples=all_triples_np,
            num_entities=len(entity2id),
            config=config
        )
        
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        fold_results.append(metrics)
        
        print(f"  AUC-ROC: {metrics['AUC-ROC']:.4f}, AUC-PR: {metrics['AUC-PR']:.4f}, "
              f"P@100: {metrics.get('P@100', 0):.4f} ({fold_time:.1f}s)")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    
    # Calculate mean and std for each metric
    metric_names = list(fold_results[0].keys())
    summary = {}
    
    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "values": values
        }
    
    # Print summary
    print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 55)
    for metric in ["AUC-ROC", "AUC-PR", "P@10", "P@50", "P@100"]:
        if metric in summary:
            s = summary[metric]
            print(f"{metric:<15} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
    
    print(f"\nTotal time: {sum(fold_times):.1f}s ({np.mean(fold_times):.1f}s per fold)")
    
    # Save detailed results
    save_cv_results(summary, fold_results, config, output_dir)
    
    # Plot results
    plot_cv_results(summary, fold_results, output_dir)
    
    return summary, fold_results


def save_cv_results(summary: dict, fold_results: list, config: dict, output_dir: str):
    """Save cross-validation results to files."""
    
    # Save summary
    with open(os.path.join(output_dir, "cv_results.txt"), "w") as f:
        f.write("10-Fold Cross-Validation Results - TriModel\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in config.items():
            if key != "device":
                f.write(f"  {key}: {value}\n")
        
        f.write("\n\nRESULTS SUMMARY:\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
        f.write("-" * 55 + "\n")
        
        for metric in ["AUC-ROC", "AUC-PR", "P@10", "P@50", "P@100", "score_gap"]:
            if metric in summary:
                s = summary[metric]
                f.write(f"{metric:<15} {s['mean']:>10.4f} {s['std']:>10.4f} "
                        f"{s['min']:>10.4f} {s['max']:>10.4f}\n")
        
        f.write("\n\nPER-FOLD RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Fold':<6} {'AUC-ROC':>10} {'AUC-PR':>10} {'P@100':>10} {'Score Gap':>12}\n")
        f.write("-" * 70 + "\n")
        
        for i, result in enumerate(fold_results, 1):
            f.write(f"{i:<6} {result['AUC-ROC']:>10.4f} {result['AUC-PR']:>10.4f} "
                    f"{result.get('P@100', 0):>10.4f} {result['score_gap']:>12.4f}\n")
    
    # Save per-fold results as CSV
    rows = []
    for i, result in enumerate(fold_results, 1):
        row = {"fold": i}
        row.update(result)
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "cv_fold_results.csv"), index=False)
    
    print(f"\nSaved results to {output_dir}/:")
    print("  - cv_results.txt")
    print("  - cv_fold_results.csv")


def plot_cv_results(summary: dict, fold_results: list, output_dir: str):
    """Plot cross-validation results."""
    figures_dir = os.path.join(output_dir, "figures")
    
    # Plot 1: Box plot of main metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ["AUC-ROC", "AUC-PR", "P@10", "P@50", "P@100"]
    data = [[r[m] for r in fold_results] for m in metrics_to_plot if m in summary]
    labels = [m for m in metrics_to_plot if m in summary]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('10-Fold Cross-Validation Results - TriModel', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.85, 1.02)
    
    # Add mean values as text
    for i, (d, label) in enumerate(zip(data, labels)):
        mean_val = np.mean(d)
        ax.annotate(f'{mean_val:.4f}', xy=(i+1, mean_val), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cv_boxplot.png"), dpi=150)
    plt.close()
    
    # Plot 2: Line plot across folds
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = list(range(1, len(fold_results) + 1))
    
    for metric, color in zip(["AUC-ROC", "AUC-PR"], ['#3498db', '#2ecc71']):
        if metric in summary:
            values = [r[metric] for r in fold_results]
            ax.plot(folds, values, 'o-', label=f'{metric} (mean: {np.mean(values):.4f})', 
                    color=color, markersize=8, linewidth=2)
            ax.axhline(y=np.mean(values), color=color, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Across Folds', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(folds)
    ax.set_ylim(0.9, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cv_folds_line.png"), dpi=150)
    plt.close()
    
    print(f"  - figures/cv_boxplot.png")
    print(f"  - figures/cv_folds_line.png")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    summary, fold_results = run_cross_validation(CONFIG)
    
    print("\n" + "=" * 70)
    print("Cross-validation complete!")
    print("=" * 70)
