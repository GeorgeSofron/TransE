"""
Enhanced Evaluation with libkge Metrics for DrugBank Knowledge Graph
Uses libkge library metrics: AUC-ROC, AUC-PR, Precision@K, Mean Rank, MRR
Evaluates both globally and per-relation.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Import libkge metrics directly (avoid full libkge init which needs bidict)
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------
# LibKGE Metrics (copied from libkge to avoid dependency issues)
# ----------------------------
def auc_roc(y_true, y_pred):
    """Compute the area under the ROC curve."""
    return roc_auc_score(y_true, y_pred)


def auc_pr(y_true, y_pred):
    """Compute the area under the precision-recall curve."""
    return average_precision_score(y_true, y_pred)


def ranks(y_true, y_pred, pos_label=1.0):
    """Compute ranks of the true labels."""
    rank_order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[rank_order]
    pos_label_mask = y_true_sorted == pos_label
    return np.nonzero(pos_label_mask)[0] + 1


def mean_rank(y_true, y_pred, pos_label=1.0):
    """Compute the mean rank of the true labels."""
    return np.mean(ranks(y_true, y_pred, pos_label=pos_label))


def mean_reciprocal_ranks(y_true, y_pred, pos_label=1.0):
    """Compute the mean reciprocal rank of the true labels."""
    return np.mean(1 / ranks(y_true, y_pred, pos_label=pos_label))


def precision_at_k(y_true, y_pred, k, pos_label=1.0):
    """Compute the precision at k."""
    if k < 1 or k > len(y_true):
        return 0.0
    rank_order = np.argsort(y_pred)[::-1]
    y_true_k_sorted = y_true[rank_order[:k]]
    return np.count_nonzero(y_true_k_sorted == pos_label) / k


def average_precision(y_true, y_pred, pos_label=1.0):
    """Compute the average precision."""
    ranks_array = ranks(y_true, y_pred, pos_label=pos_label)
    pk_list = [precision_at_k(y_true, y_pred, k, pos_label=pos_label) for k in ranks_array]
    return np.mean(pk_list) if pk_list else 0.0

from model import TransE, ComplEx, TriModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_path: str, device: str = "cpu"):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    num_entities = checkpoint["num_entities"]
    num_relations = checkpoint["num_relations"]
    embedding_dim = checkpoint["embedding_dim"]
    entity2id = checkpoint["entity2id"]
    relation2id = checkpoint["relation2id"]
    
    # Detect model type
    if "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
    elif "p_norm" in checkpoint:
        model_type = "TransE"
    else:
        state_dict = checkpoint["model_state_dict"]
        if any("v1" in key for key in state_dict.keys()):
            model_type = "TriModel"
        elif any("ent_re" in key for key in state_dict.keys()):
            model_type = "ComplEx"
        else:
            model_type = "TransE"
    
    # Create model
    if model_type == "TransE":
        p_norm = checkpoint.get("p_norm", 1)
        model = TransE(num_entities, num_relations, dim=embedding_dim, p_norm=p_norm)
    elif model_type == "ComplEx":
        model = ComplEx(num_entities, num_relations, dim=embedding_dim)
    elif model_type == "TriModel":
        model = TriModel(num_entities, num_relations, dim=embedding_dim)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, entity2id, relation2id, model_type


def load_triples(path: str, entity2id: dict, relation2id: dict) -> np.ndarray:
    """Load and encode triples from file."""
    df = pd.read_csv(path, sep="\t", header=None, names=["h", "r", "t"])
    
    # Filter to known entities/relations
    df = df[
        (df["h"].isin(entity2id.keys())) &
        (df["r"].isin(relation2id.keys())) &
        (df["t"].isin(entity2id.keys()))
    ].reset_index(drop=True)
    
    h = df["h"].map(entity2id).to_numpy()
    r = df["r"].map(relation2id).to_numpy()
    t = df["t"].map(entity2id).to_numpy()
    
    return np.stack([h, r, t], axis=1), df


def generate_negatives(pos_triples: np.ndarray, num_entities: int, 
                       true_triples_set: set, ratio: int = 1) -> np.ndarray:
    """Generate negative triples by corrupting head or tail."""
    neg_list = []
    
    for h, r, t in pos_triples:
        for _ in range(ratio):
            for attempt in range(10):
                if random.random() < 0.5:
                    # Corrupt head
                    new_h = random.randint(0, num_entities - 1)
                    candidate = (new_h, r, t)
                    if candidate not in true_triples_set:
                        neg_list.append([new_h, r, t])
                        break
                else:
                    # Corrupt tail
                    new_t = random.randint(0, num_entities - 1)
                    candidate = (h, r, new_t)
                    if candidate not in true_triples_set:
                        neg_list.append([h, r, new_t])
                        break
    
    return np.array(neg_list)


@torch.no_grad()
def compute_scores(model, triples: np.ndarray, model_type: str, device: str) -> np.ndarray:
    """Compute scores for triples."""
    model.eval()
    triples_tensor = torch.tensor(triples, dtype=torch.long).to(device)
    
    if model_type == "TransE":
        # TransE: lower distance = better, so negate for consistency
        scores = -model(triples_tensor).cpu().numpy()
    else:
        # ComplEx/TriModel: higher score = better
        scores = model(triples_tensor).cpu().numpy()
    
    return scores


def evaluate_global(pos_scores: np.ndarray, neg_scores: np.ndarray) -> dict:
    """Compute global classification and ranking metrics using libkge."""
    # Labels: 1 for positive, 0 for negative
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    metrics = {
        "AUC-ROC": auc_roc(y_true, y_scores),
        "AUC-PR": auc_pr(y_true, y_scores),
        "AP": average_precision(y_true, y_scores),
        "MRR": mean_reciprocal_ranks(y_true, y_scores),
        "Mean Rank": mean_rank(y_true, y_scores),
        "P@10": precision_at_k(y_true, y_scores, k=10),
        "P@50": precision_at_k(y_true, y_scores, k=50),
        "P@100": precision_at_k(y_true, y_scores, k=100),
    }
    
    return metrics


def evaluate_per_relation(test_triples: np.ndarray, neg_triples: np.ndarray,
                          pos_scores: np.ndarray, neg_scores: np.ndarray,
                          relation2id: dict) -> dict:
    """Compute metrics per relation type."""
    id2relation = {v: k for k, v in relation2id.items()}
    
    metrics_per_rel = {}
    
    for rel_id in sorted(set(test_triples[:, 1])):
        rel_name = id2relation[rel_id]
        
        # Get indices for this relation
        pos_mask = test_triples[:, 1] == rel_id
        neg_mask = neg_triples[:, 1] == rel_id
        
        if pos_mask.sum() < 5 or neg_mask.sum() < 5:
            continue  # Skip relations with too few samples
        
        rel_pos_scores = pos_scores[pos_mask]
        rel_neg_scores = neg_scores[neg_mask]
        
        y_true = np.concatenate([np.ones(len(rel_pos_scores)), np.zeros(len(rel_neg_scores))])
        y_scores = np.concatenate([rel_pos_scores, rel_neg_scores])
        
        k = min(50, len(y_true) - 1)
        
        metrics_per_rel[rel_name] = {
            "count": int(pos_mask.sum()),
            "AUC-ROC": auc_roc(y_true, y_scores),
            "AUC-PR": auc_pr(y_true, y_scores),
            "MRR": mean_reciprocal_ranks(y_true, y_scores),
            "P@k": precision_at_k(y_true, y_scores, k=k) if k > 0 else 0,
        }
    
    return metrics_per_rel


def plot_metrics_comparison(metrics_per_rel: dict, output_dir: str, metric_name: str = "AUC-ROC"):
    """Plot per-relation metrics as bar chart."""
    relations = list(metrics_per_rel.keys())
    values = [metrics_per_rel[r][metric_name] for r in relations]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(relations)), values, color='steelblue')
    plt.xlabel('Relation', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} per Relation', fontsize=14)
    plt.xticks(range(len(relations)), relations, rotation=45, ha='right')
    plt.axhline(y=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"per_relation_{metric_name.lower().replace('-', '_')}.png"), dpi=150)
    plt.close()


def save_results(global_metrics: dict, per_rel_metrics: dict, 
                 model_type: str, output_dir: str):
    """Save all metrics to files."""
    
    # Save global metrics
    with open(os.path.join(output_dir, "libkge_metrics.txt"), "w") as f:
        f.write(f"LibKGE Evaluation Metrics - {model_type}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GLOBAL METRICS:\n")
        f.write("-" * 40 + "\n")
        for name, value in global_metrics.items():
            f.write(f"  {name:20s}: {value:.4f}\n")
        
        f.write("\n\nPER-RELATION METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Relation':30s} {'Count':>6s} {'AUC-ROC':>10s} {'AUC-PR':>10s} {'MRR':>10s}\n")
        f.write("-" * 60 + "\n")
        
        for rel, metrics in sorted(per_rel_metrics.items()):
            f.write(f"{rel:30s} {metrics['count']:>6d} {metrics['AUC-ROC']:>10.4f} "
                    f"{metrics['AUC-PR']:>10.4f} {metrics['MRR']:>10.4f}\n")
        
        # Averages
        avg_auc_roc = np.mean([m["AUC-ROC"] for m in per_rel_metrics.values()])
        avg_auc_pr = np.mean([m["AUC-PR"] for m in per_rel_metrics.values()])
        avg_mrr = np.mean([m["MRR"] for m in per_rel_metrics.values()])
        
        f.write("-" * 60 + "\n")
        f.write(f"{'AVERAGE':30s} {'':>6s} {avg_auc_roc:>10.4f} {avg_auc_pr:>10.4f} {avg_mrr:>10.4f}\n")
    
    # Save per-relation metrics as CSV
    rows = []
    for rel, metrics in per_rel_metrics.items():
        row = {"relation": rel}
        row.update(metrics)
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "per_relation_metrics.csv"), index=False)
    
    print(f"\nSaved to {output_dir}/:")
    print("  - libkge_metrics.txt")
    print("  - per_relation_metrics.csv")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    
    # Configuration - choose model
    MODEL_PATH = "outputs_transe/transe_model.pt"
    DATA_DIR = "data/transe"
    
    #MODEL_PATH = "outputs_complex/complex_model.pt"
    #DATA_DIR = "data/complex"
    
    #MODEL_PATH = "outputs_trimodel/trimodel_model.pt"
    #DATA_DIR = "data/trimodel"
    
    DEVICE = "cpu"
    NEG_RATIO = 1  # Number of negatives per positive
    
    # Derive output directory from model path
    OUTPUT_DIR = os.path.dirname(MODEL_PATH)
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("=" * 60)
    print("LibKGE Evaluation for DrugBank Knowledge Graph")
    print("=" * 60)
    
    print(f"\nLoading model: {MODEL_PATH}")
    model, entity2id, relation2id, model_type = load_model(MODEL_PATH, DEVICE)
    print(f"  Model type: {model_type}")
    print(f"  Entities: {len(entity2id):,}")
    print(f"  Relations: {len(relation2id)}")
    
    # Load data
    train_path = os.path.join(DATA_DIR, "train.txt")
    test_path = os.path.join(DATA_DIR, "test.txt")
    
    print(f"\nLoading triples...")
    train_triples, _ = load_triples(train_path, entity2id, relation2id)
    test_triples, test_df = load_triples(test_path, entity2id, relation2id)
    print(f"  Train: {len(train_triples):,}")
    print(f"  Test:  {len(test_triples):,}")
    
    # Build true triples set for filtered negatives
    all_triples = np.concatenate([train_triples, test_triples])
    true_triples_set = set(map(tuple, all_triples.tolist()))
    
    # Generate negatives
    print(f"\nGenerating negative triples (ratio 1:{NEG_RATIO})...")
    neg_triples = generate_negatives(test_triples, len(entity2id), true_triples_set, ratio=NEG_RATIO)
    print(f"  Negative triples: {len(neg_triples):,}")
    
    # Compute scores
    print("\nComputing scores...")
    pos_scores = compute_scores(model, test_triples, model_type, DEVICE)
    neg_scores = compute_scores(model, neg_triples, model_type, DEVICE)
    
    print(f"  Positive scores - mean: {pos_scores.mean():.4f}, std: {pos_scores.std():.4f}")
    print(f"  Negative scores - mean: {neg_scores.mean():.4f}, std: {neg_scores.std():.4f}")
    
    # Global evaluation using libkge metrics
    print("\nComputing global metrics (libkge)...")
    global_metrics = evaluate_global(pos_scores, neg_scores)
    
    print("\n" + "-" * 40)
    print("GLOBAL METRICS:")
    print("-" * 40)
    for name, value in global_metrics.items():
        print(f"  {name:15s}: {value:.4f}")
    
    # Per-relation evaluation
    print("\nComputing per-relation metrics...")
    per_rel_metrics = evaluate_per_relation(test_triples, neg_triples, 
                                            pos_scores, neg_scores, relation2id)
    
    print("\n" + "-" * 60)
    print("PER-RELATION METRICS:")
    print("-" * 60)
    print(f"{'Relation':30s} {'Count':>6s} {'AUC-ROC':>10s} {'AUC-PR':>10s}")
    print("-" * 60)
    for rel, metrics in sorted(per_rel_metrics.items()):
        print(f"{rel:30s} {metrics['count']:>6d} {metrics['AUC-ROC']:>10.4f} {metrics['AUC-PR']:>10.4f}")
    
    # Plot per-relation metrics
    print("\nGenerating plots...")
    plot_metrics_comparison(per_rel_metrics, FIGURES_DIR, "AUC-ROC")
    plot_metrics_comparison(per_rel_metrics, FIGURES_DIR, "AUC-PR")
    print(f"  Saved: {FIGURES_DIR}/per_relation_auc_roc.png")
    print(f"  Saved: {FIGURES_DIR}/per_relation_auc_pr.png")
    
    # Save all results
    save_results(global_metrics, per_rel_metrics, model_type, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
