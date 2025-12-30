"""
ROC Curve Computation for Knowledge Graph Embedding Models
Computes ROC curve and AUC by classifying true vs corrupted triples.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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
    
    # Detect model type from checkpoint
    if "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
    elif "p_norm" in checkpoint:
        model_type = "TransE"
    else:
        # Check state dict keys for model type detection
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, entity2id, relation2id, model_type


def load_triples(path: str, entity2id: dict, relation2id: dict) -> torch.Tensor:
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
    
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


def corrupt_triples(pos_triples: torch.Tensor, num_entities: int, 
                    true_triples_set: set = None) -> torch.Tensor:
    """Generate negative triples by corrupting head or tail."""
    neg = pos_triples.clone()
    batch_size = neg.size(0)
    
    mask = torch.rand(batch_size) < 0.5  # True => corrupt head
    
    for i in range(batch_size):
        h, r, t = neg[i].tolist()
        
        for _ in range(10):  # max attempts
            random_entity = torch.randint(0, num_entities, (1,)).item()
            
            if mask[i]:
                candidate = (random_entity, r, t)
            else:
                candidate = (h, r, random_entity)
            
            if true_triples_set is None or candidate not in true_triples_set:
                if mask[i]:
                    neg[i, 0] = random_entity
                else:
                    neg[i, 2] = random_entity
                break
    
    return neg


@torch.no_grad()
def compute_scores(model, triples: torch.Tensor, model_type: str, device: str) -> np.ndarray:
    """
    Compute scores for triples.
    For TransE: lower distance = more likely true (convert to higher = better)
    For ComplEx/TriModel: higher score = more likely true
    """
    model.eval()
    triples = triples.to(device)
    
    if model_type == "TransE":
        # TransE returns distance - lower is better, so negate
        scores = -model(triples).cpu().numpy()
    else:
        # ComplEx/TriModel return scores - higher is better
        scores = model(triples).cpu().numpy()
    
    return scores


def compute_roc_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray):
    """Compute ROC curve, AUC, and PR curve."""
    # Labels: 1 for positive, 0 for negative
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }


def plot_roc_curve(metrics: dict, model_type: str, output_dir: str):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["fpr"], metrics["tpr"], 
             color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_type} - ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/roc_curve.png")


def plot_pr_curve(metrics: dict, model_type: str, output_dir: str):
    """Plot and save Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(metrics["recall"], metrics["precision"], 
             color='darkorange', lw=2, 
             label=f'PR curve (AP = {metrics["pr_auc"]:.4f})')
    plt.axhline(y=0.5, color='navy', lw=2, linestyle='--', label='Random (balanced)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{model_type} - Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/pr_curve.png")


def plot_score_distribution(pos_scores: np.ndarray, neg_scores: np.ndarray, 
                            model_type: str, output_dir: str):
    """Plot score distributions for positive vs negative triples."""
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=50, alpha=0.6, label='Positive (True)', color='green', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.6, label='Negative (Corrupted)', color='red', density=True)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'{model_type} - Score Distribution', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/score_distribution.png")


def save_metrics(metrics: dict, model_type: str, output_dir: str):
    """Save metrics to text file."""
    output_path = os.path.join(output_dir, "roc_metrics.txt")
    with open(output_path, "w") as f:
        f.write(f"ROC Metrics - {model_type}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"ROC AUC:           {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC (AP):       {metrics['pr_auc']:.4f}\n")
    print(f"  Saved: {output_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    
    # Configuration - choose model
    #MODEL_PATH = "outputs_transe/transe_model.pt"
    #DATA_DIR = "data/transe"
    
    #MODEL_PATH = "outputs_complex/complex_model.pt"
    #DATA_DIR = "data/complex"
    
    MODEL_PATH = "outputs_trimodel/trimodel_model.pt"
    DATA_DIR = "data/trimodel"
    
    DEVICE = "cpu"
    
    # Derive output directory from model path
    OUTPUT_DIR = os.path.dirname(MODEL_PATH)
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    
    print(f"Loading model: {MODEL_PATH}")
    model, entity2id, relation2id, model_type = load_model(MODEL_PATH, DEVICE)
    print(f"  Model type: {model_type}")
    print(f"  Entities: {len(entity2id):,}")
    print(f"  Relations: {len(relation2id)}")
    
    # Load test triples
    test_path = os.path.join(DATA_DIR, "test.txt")
    print(f"\nLoading test triples: {test_path}")
    test_triples = load_triples(test_path, entity2id, relation2id)
    print(f"  Test triples: {len(test_triples):,}")
    
    # Load train triples for filtering
    train_path = os.path.join(DATA_DIR, "train.txt")
    train_triples = load_triples(train_path, entity2id, relation2id)
    all_triples = torch.cat([train_triples, test_triples], dim=0)
    true_triples_set = set(map(tuple, all_triples.numpy().tolist()))
    
    # Generate negative triples (same size as positive)
    print("\nGenerating negative triples...")
    neg_triples = corrupt_triples(test_triples, len(entity2id), true_triples_set)
    print(f"  Negative triples: {len(neg_triples):,}")
    
    # Compute scores
    print("\nComputing scores...")
    pos_scores = compute_scores(model, test_triples, model_type, DEVICE)
    neg_scores = compute_scores(model, neg_triples, model_type, DEVICE)
    
    print(f"  Positive scores - mean: {pos_scores.mean():.4f}, std: {pos_scores.std():.4f}")
    print(f"  Negative scores - mean: {neg_scores.mean():.4f}, std: {neg_scores.std():.4f}")
    
    # Compute ROC metrics
    print("\nComputing ROC metrics...")
    metrics = compute_roc_metrics(pos_scores, neg_scores)
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR AUC:  {metrics['pr_auc']:.4f}")
    
    # Plot and save
    print("\nSaving plots...")
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    plot_roc_curve(metrics, model_type, figures_dir)
    plot_pr_curve(metrics, model_type, figures_dir)
    plot_score_distribution(pos_scores, neg_scores, model_type, figures_dir)
    save_metrics(metrics, model_type, OUTPUT_DIR)
    
    print("\nDone!")
