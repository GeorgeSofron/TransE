"""
Evaluate DrugBank-Trained TransE Model on UniProt Dataset
==========================================================
This script evaluates the trained TransE model on external drug-target
interactions from UniProt to test generalization ability.

Prerequisites:
- Run convert_uniprot_to_drugbank.py first to generate data/uniprot_filtered.txt
- Trained model must exist at outputs/transe_model.pt

Output:
- outputs/uniprot_evaluation.txt: Evaluation results
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import TransE  # Shared model definition


# ----------------------------
# Load trained model
# ----------------------------
def load_model(checkpoint_path, device="cpu"):
    """Load trained TransE model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = TransE(
        num_entities=ckpt["num_entities"],
        num_relations=ckpt["num_relations"],
        dim=ckpt["embedding_dim"],
        p_norm=ckpt["p_norm"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["entity2id"], ckpt["relation2id"]


# ----------------------------
# Load triples
# ----------------------------
def load_triples(path):
    """Load triples from tab-separated file."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )


def triples_to_id_set(df, entity2id, relation2id):
    """Convert DataFrame of triples to set of ID tuples."""
    return set(
        zip(
            df["source"].map(entity2id),
            df["relation"].map(relation2id),
            df["target"].map(entity2id),
        )
    )


def encode_triples(df, entity2id, relation2id):
    """Encode triples DataFrame to tensor of IDs."""
    h = df["source"].map(entity2id).to_numpy()
    r = df["relation"].map(relation2id).to_numpy()
    t = df["target"].map(entity2id).to_numpy()
    triples = np.stack([h, r, t], axis=1)
    return torch.tensor(triples, dtype=torch.long)


# ----------------------------
# Ranking functions (filtered) - Optimized with vectorization
# ----------------------------
@torch.no_grad()
def compute_ranks_batch(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Compute filtered ranks for a batch of test triples.
    
    Args:
        model: Trained TransE model
        test_triples: Tensor of shape (N, 3) with [head, relation, tail] IDs
        all_true: Set of all true triples for filtering
        num_entities: Total number of entities
        device: torch device
        batch_size: Number of triples to evaluate at once
        
    Returns:
        Tuple of (head_ranks, tail_ranks) arrays
    """
    model.eval()
    n_test = test_triples.size(0)
    
    all_tail_ranks = []
    all_head_ranks = []
    
    # Pre-compute all entity embeddings
    all_ent_emb = model.ent.weight.data  # (num_entities, dim)
    
    for start in tqdm(range(0, n_test, batch_size), desc="Evaluating batches"):
        batch = test_triples[start:start + batch_size].to(device)
        batch_h = batch[:, 0]
        batch_r = batch[:, 1]
        batch_t = batch[:, 2]
        
        h_emb = model.ent(batch_h)  # (batch, dim)
        r_emb = model.rel(batch_r)  # (batch, dim)
        t_emb = model.ent(batch_t)  # (batch, dim)
        
        # --- Tail prediction: score all entities as potential tails ---
        hr = h_emb + r_emb  # (batch, dim)
        tail_scores = torch.cdist(hr.unsqueeze(1), all_ent_emb.unsqueeze(0), p=model.p_norm).squeeze(1)
        tail_scores = tail_scores.cpu().numpy()
        
        # --- Head prediction: score all entities as potential heads ---
        t_minus_r = t_emb - r_emb  # (batch, dim)
        head_scores = torch.cdist(t_minus_r.unsqueeze(1), all_ent_emb.unsqueeze(0), p=model.p_norm).squeeze(1)
        head_scores = head_scores.cpu().numpy()
        
        # Apply filtering and compute ranks
        for i in range(batch.size(0)):
            h, r, t = batch[i].cpu().tolist()
            
            # Filter tail scores
            tail_sc = tail_scores[i].copy()
            for e in range(num_entities):
                if (h, r, e) in all_true and e != t:
                    tail_sc[e] = np.inf
            tail_rank = int((tail_sc < tail_sc[t]).sum() + 1)
            all_tail_ranks.append(tail_rank)
            
            # Filter head scores
            head_sc = head_scores[i].copy()
            for e in range(num_entities):
                if (e, r, t) in all_true and e != h:
                    head_sc[e] = np.inf
            head_rank = int((head_sc < head_sc[h]).sum() + 1)
            all_head_ranks.append(head_rank)
    
    return np.array(all_head_ranks), np.array(all_tail_ranks)


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, test_triples, all_true, num_entities, device, batch_size=256):
    """
    Evaluate model on test triples using batched computation.
    
    Returns:
        Dictionary with MRR, Hits@1, Hits@3, Hits@10 and separate head/tail metrics
    """
    head_ranks, tail_ranks = compute_ranks_batch(
        model, test_triples, all_true, num_entities, device, batch_size
    )
    
    # Combine head and tail ranks
    all_ranks = np.concatenate([head_ranks, tail_ranks])
    
    return {
        # Overall metrics
        "MRR": float(np.mean(1.0 / all_ranks)),
        "Hits@1": float((all_ranks <= 1).mean()),
        "Hits@3": float((all_ranks <= 3).mean()),
        "Hits@10": float((all_ranks <= 10).mean()),
        # Head prediction metrics (predict drug given target)
        "Head_MRR": float(np.mean(1.0 / head_ranks)),
        "Head_Hits@1": float((head_ranks <= 1).mean()),
        "Head_Hits@10": float((head_ranks <= 10).mean()),
        # Tail prediction metrics (predict target given drug)
        "Tail_MRR": float(np.mean(1.0 / tail_ranks)),
        "Tail_Hits@1": float((tail_ranks <= 1).mean()),
        "Tail_Hits@10": float((tail_ranks <= 10).mean()),
        # Rank statistics
        "Mean_Rank": float(np.mean(all_ranks)),
        "Median_Rank": float(np.median(all_ranks)),
    }


def save_results(metrics, output_path):
    """Save evaluation results to file."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("UniProt External Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  MRR:      {metrics['MRR']:.4f}\n")
        f.write(f"  Hits@1:   {metrics['Hits@1']:.4f}\n")
        f.write(f"  Hits@3:   {metrics['Hits@3']:.4f}\n")
        f.write(f"  Hits@10:  {metrics['Hits@10']:.4f}\n\n")
        
        f.write("Head Prediction (Drug given Target):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  MRR:      {metrics['Head_MRR']:.4f}\n")
        f.write(f"  Hits@1:   {metrics['Head_Hits@1']:.4f}\n")
        f.write(f"  Hits@10:  {metrics['Head_Hits@10']:.4f}\n\n")
        
        f.write("Tail Prediction (Target given Drug):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  MRR:      {metrics['Tail_MRR']:.4f}\n")
        f.write(f"  Hits@1:   {metrics['Tail_Hits@1']:.4f}\n")
        f.write(f"  Hits@10:  {metrics['Tail_Hits@10']:.4f}\n\n")
        
        f.write("Rank Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean Rank:   {metrics['Mean_Rank']:.1f}\n")
        f.write(f"  Median Rank: {metrics['Median_Rank']:.1f}\n")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Configuration
    DEVICE = "cpu"  # change to "cuda" if available
    BATCH_SIZE = 256
    
    MODEL_PATH = "outputs/transe_model.pt"
    TRAIN_PATH = "data/train.txt"
    UNIPROT_PATH = "data/uniprot_filtered.txt"
    OUTPUT_PATH = "outputs/uniprot_evaluation.txt"
    
    print("=" * 60)
    print("UniProt External Evaluation")
    print("=" * 60)
    
    # Check if UniProt filtered file exists
    if not os.path.exists(UNIPROT_PATH):
        print(f"\nError: {UNIPROT_PATH} not found!")
        print("Please run convert_uniprot_to_drugbank.py first.")
        exit(1)
    
    # Load model
    print("\n[1/4] Loading trained model...")
    model, entity2id, relation2id = load_model(MODEL_PATH, DEVICE)
    num_entities = len(entity2id)
    print(f"Model loaded: {num_entities} entities, {len(relation2id)} relations")
    
    # Load training data (for filtering known triples)
    print("\n[2/4] Loading training data...")
    train_df = load_triples(TRAIN_PATH)
    print(f"Training triples: {len(train_df)}")
    
    # Load UniProt test data
    print("\n[3/4] Loading UniProt test data...")
    uniprot_df = load_triples(UNIPROT_PATH)
    print(f"UniProt triples (before filtering): {len(uniprot_df)}")
    
    # Filter to entities/relations known to model
    uniprot_df = uniprot_df[
        (uniprot_df["source"].isin(entity2id.keys())) &
        (uniprot_df["relation"].isin(relation2id.keys())) &
        (uniprot_df["target"].isin(entity2id.keys()))
    ].reset_index(drop=True)
    
    print(f"UniProt triples (after filtering): {len(uniprot_df)}")
    print(f"Unique drugs: {uniprot_df['source'].nunique()}")
    print(f"Unique proteins: {uniprot_df['target'].nunique()}")
    
    # Check overlap with training data
    train_set = set(zip(train_df['source'], train_df['relation'], train_df['target']))
    uniprot_set = set(zip(uniprot_df['source'], uniprot_df['relation'], uniprot_df['target']))
    overlap = train_set & uniprot_set
    novel = uniprot_set - train_set
    
    print(f"\nOverlap with training data: {len(overlap)} triples")
    print(f"Novel triples (not in training): {len(novel)} triples")
    
    # Encode triples
    test_triples = encode_triples(uniprot_df, entity2id, relation2id)
    
    # Build set of all true triples for filtering
    all_true = triples_to_id_set(train_df, entity2id, relation2id)
    all_true |= triples_to_id_set(uniprot_df, entity2id, relation2id)
    
    # Evaluate
    print("\n[4/4] Evaluating model on UniProt data...")
    metrics = evaluate(
        model,
        test_triples,
        all_true,
        num_entities,
        DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (UniProt External Dataset)")
    print("=" * 60)
    
    print("\nOverall Metrics:")
    print(f"  MRR:      {metrics['MRR']:.4f}")
    print(f"  Hits@1:   {metrics['Hits@1']:.4f}")
    print(f"  Hits@3:   {metrics['Hits@3']:.4f}")
    print(f"  Hits@10:  {metrics['Hits@10']:.4f}")
    
    print("\nHead Prediction (Drug given Target):")
    print(f"  MRR:      {metrics['Head_MRR']:.4f}")
    print(f"  Hits@1:   {metrics['Head_Hits@1']:.4f}")
    print(f"  Hits@10:  {metrics['Head_Hits@10']:.4f}")
    
    print("\nTail Prediction (Target given Drug):")
    print(f"  MRR:      {metrics['Tail_MRR']:.4f}")
    print(f"  Hits@1:   {metrics['Tail_Hits@1']:.4f}")
    print(f"  Hits@10:  {metrics['Tail_Hits@10']:.4f}")
    
    print("\nRank Statistics:")
    print(f"  Mean Rank:   {metrics['Mean_Rank']:.1f}")
    print(f"  Median Rank: {metrics['Median_Rank']:.1f}")
    
    # Save results
    save_results(metrics, OUTPUT_PATH)
    print(f"\nResults saved to: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
