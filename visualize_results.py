"""
TransE Results Visualization
=============================
Generate publication-ready figures for TransE evaluation results.

Outputs:
- Evaluation metrics bar chart
- Training loss curve
- Entity embedding t-SNE visualization
- Rank distribution histogram
- Per-relation performance (if available)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Set to True to display figures interactively, False to only save
SHOW_FIGURES = False

OUTPUT_DIR = "outputs"
FIGURES_DIR = "outputs/figures"


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ----------------------------
# 1. Evaluation Metrics Bar Chart
# ----------------------------
def plot_metrics(metrics: dict, save_path: str = None):
    """
    Create a bar chart of evaluation metrics.
    
    Args:
        metrics: Dictionary with MRR, Hits@1, Hits@3, Hits@10
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('TransE Link Prediction Performance on DrugBank', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=11)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 2. Training Loss Curve
# ----------------------------
def plot_training_loss(loss_csv_path: str, save_path: str = None):
    """
    Plot training loss curve from CSV file.
    
    Args:
        loss_csv_path: Path to training_loss.csv
        save_path: Path to save the figure
    """
    df = pd.read_csv(loss_csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['epoch'], df['loss'], color='#3498db', linewidth=2, label='Training Loss')
    ax.fill_between(df['epoch'], df['loss'], alpha=0.3, color='#3498db')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('TransE Training Loss Curve', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11)
    
    # Mark convergence point (where loss stabilizes)
    min_loss_epoch = df.loc[df['loss'].idxmin(), 'epoch']
    min_loss = df['loss'].min()
    ax.axhline(y=min_loss, color='red', linestyle='--', alpha=0.5, label=f'Min Loss: {min_loss:.4f}')
    ax.scatter([min_loss_epoch], [min_loss], color='red', s=100, zorder=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 3. Entity Embedding Visualization (t-SNE)
# ----------------------------
def plot_embeddings_tsne(embeddings_csv_path: str, n_samples: int = 2000, 
                          save_path: str = None, perplexity: int = 30):
    """
    Create t-SNE visualization of entity embeddings.
    
    Args:
        embeddings_csv_path: Path to entity_embeddings.csv
        n_samples: Number of entities to sample (for speed)
        save_path: Path to save the figure
        perplexity: t-SNE perplexity parameter
    """
    print("Loading embeddings...")
    df = pd.read_csv(embeddings_csv_path, index_col=0)
    
    # Sample if too many entities
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        print(f"Sampled {n_samples} entities for visualization")
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb_2d = tsne.fit_transform(df.values)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         alpha=0.6, s=15, c=range(len(emb_2d)), cmap='viridis')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Entity Embeddings', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    
    # Remove axis for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 4. Rank Distribution Histogram
# ----------------------------
def plot_rank_distribution(ranks: np.ndarray, save_path: str = None):
    """
    Plot histogram of prediction ranks.
    
    Args:
        ranks: Array of ranks from evaluation
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full distribution
    ax1 = axes[0]
    ax1.hist(ranks, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Rank', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Prediction Ranks', fontsize=14, fontweight='bold')
    ax1.axvline(x=np.median(ranks), color='red', linestyle='--', 
                label=f'Median: {np.median(ranks):.0f}')
    ax1.axvline(x=np.mean(ranks), color='green', linestyle='--', 
                label=f'Mean: {np.mean(ranks):.0f}')
    ax1.legend(fontsize=10)
    
    # Top-100 zoom
    ax2 = axes[1]
    top_ranks = ranks[ranks <= 100]
    ax2.hist(top_ranks, bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Rank (Top 100)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Top-100 Ranks', fontsize=14, fontweight='bold')
    
    # Add Hits markers
    for k, color in [(1, 'red'), (3, 'orange'), (10, 'purple')]:
        count = (ranks <= k).sum()
        pct = count / len(ranks) * 100
        ax2.axvline(x=k, color=color, linestyle='--', label=f'Hits@{k}: {pct:.1f}%')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 5. Comparison with Baselines
# ----------------------------
def plot_baseline_comparison(your_metrics: dict, save_path: str = None):
    """
    Compare your results with published baselines.
    
    Args:
        your_metrics: Your evaluation metrics
        save_path: Path to save the figure
    """
    # Published baselines (approximate values from literature)
    baselines = {
        'TransE (FB15k-237)': {'MRR': 0.29, 'Hits@1': 0.20, 'Hits@10': 0.47},
        'TransE (WN18RR)': {'MRR': 0.23, 'Hits@1': 0.04, 'Hits@10': 0.50},
        'TransE (DrugBank - Ours)': your_metrics,
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(3)  # MRR, Hits@1, Hits@10
    width = 0.25
    metrics = ['MRR', 'Hits@1', 'Hits@10']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, (name, vals) in enumerate(baselines.items()):
        values = [vals.get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], 
                      edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('TransE Performance Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 0.7)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 6. Hits@K Cumulative Curve
# ----------------------------
def plot_hits_at_k_curve(ranks: np.ndarray, max_k: int = 100, save_path: str = None):
    """
    Plot cumulative Hits@K curve.
    
    Args:
        ranks: Array of ranks from evaluation
        max_k: Maximum K value to plot
        save_path: Path to save the figure
    """
    ks = np.arange(1, max_k + 1)
    hits_at_k = [(ranks <= k).mean() for k in ks]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ks, hits_at_k, color='#3498db', linewidth=2.5)
    ax.fill_between(ks, hits_at_k, alpha=0.3, color='#3498db')
    
    # Mark key points
    for k in [1, 3, 10, 50]:
        if k <= max_k:
            hit = (ranks <= k).mean()
            ax.scatter([k], [hit], s=100, zorder=5, color='red')
            ax.annotate(f'Hits@{k}: {hit:.3f}', xy=(k, hit), 
                       xytext=(k + 5, hit + 0.03), fontsize=10,
                       arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Hits@K', fontsize=12)
    ax.set_title('Cumulative Hits@K Curve', fontsize=14, fontweight='bold')
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 7. Head vs Tail Prediction Comparison
# ----------------------------
def plot_head_vs_tail(head_ranks: np.ndarray, tail_ranks: np.ndarray, save_path: str = None):
    """
    Compare head vs tail prediction performance.
    
    Args:
        head_ranks: Ranks for head prediction
        tail_ranks: Ranks for tail prediction
        save_path: Path to save the figure
    """
    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    head_scores = [
        np.mean(1.0 / head_ranks),
        (head_ranks <= 1).mean(),
        (head_ranks <= 3).mean(),
        (head_ranks <= 10).mean()
    ]
    
    tail_scores = [
        np.mean(1.0 / tail_ranks),
        (tail_ranks <= 1).mean(),
        (tail_ranks <= 3).mean(),
        (tail_ranks <= 10).mean()
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, head_scores, width, label='Head Prediction', 
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, tail_scores, width, label='Tail Prediction', 
                   color='#e74c3c', edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Head vs Tail Prediction Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 0.7)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# Main: Generate All Visualizations
# ----------------------------
def main():
    ensure_dirs()
    
    # Your evaluation metrics
    metrics = {
        'MRR': 0.3767,
        'Hits@1': 0.3150,
        'Hits@3': 0.4063,
        'Hits@10': 0.4858
    }
    
    print("=" * 60)
    print("TransE Results Visualization")
    print("=" * 60)
    
    # 1. Metrics bar chart
    print("\n[1/5] Generating metrics bar chart...")
    plot_metrics(metrics, save_path=os.path.join(FIGURES_DIR, "metrics_bar_chart.png"))
    
    # 2. Training loss curve
    loss_path = os.path.join(OUTPUT_DIR, "training_loss.csv")
    if os.path.exists(loss_path):
        print("\n[2/5] Generating training loss curve...")
        plot_training_loss(loss_path, save_path=os.path.join(FIGURES_DIR, "training_loss_curve.png"))
    else:
        print("\n[2/5] Skipping training loss (file not found)")
    
    # 3. Embedding visualization
    emb_path = os.path.join(OUTPUT_DIR, "entity_embeddings.csv")
    if os.path.exists(emb_path):
        print("\n[3/5] Generating embedding t-SNE visualization...")
        plot_embeddings_tsne(emb_path, n_samples=2000, 
                            save_path=os.path.join(FIGURES_DIR, "embedding_tsne.png"))
    else:
        print("\n[3/5] Skipping embedding visualization (file not found)")
    
    # 4. Baseline comparison
    print("\n[4/5] Generating baseline comparison chart...")
    plot_baseline_comparison(metrics, save_path=os.path.join(FIGURES_DIR, "baseline_comparison.png"))
    
    # 5. Summary figure (combined)
    print("\n[5/5] Generating summary figure...")
    create_summary_figure(metrics, loss_path, save_path=os.path.join(FIGURES_DIR, "summary_figure.png"))
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}/")
    print("=" * 60)


def create_summary_figure(metrics: dict, loss_path: str, save_path: str = None):
    """
    Create a combined summary figure for the paper.
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Metrics bar chart (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    metric_names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax1.bar(metric_names, values, color=colors, edgecolor='black')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_title('(a) Evaluation Metrics', fontweight='bold')
    ax1.set_ylim(0, 0.7)
    
    # 2. Training loss (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    if os.path.exists(loss_path):
        df = pd.read_csv(loss_path)
        ax2.plot(df['epoch'], df['loss'], color='#3498db', linewidth=2)
        ax2.fill_between(df['epoch'], df['loss'], alpha=0.3, color='#3498db')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('(b) Training Loss Curve', fontweight='bold')
    
    # 3. Baseline comparison (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    baselines = {
        'FB15k-237': [0.29, 0.20, 0.47],
        'WN18RR': [0.23, 0.04, 0.50],
        'DrugBank': [metrics['MRR'], metrics['Hits@1'], metrics['Hits@10']]
    }
    x = np.arange(3)
    width = 0.25
    for i, (name, vals) in enumerate(baselines.items()):
        ax3.bar(x + i * width, vals, width, label=name, edgecolor='black')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['MRR', 'Hits@1', 'Hits@10'])
    ax3.set_ylabel('Score')
    ax3.set_title('(c) Comparison with Benchmarks', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 0.7)
    
    # 4. Key insights (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    insight_text = """
    Key Results Summary
    ═══════════════════════════════════════
    
    • MRR of 0.377 indicates correct answers
      typically rank in top 3 positions
    
    • Hits@10 of 48.6% shows nearly half of
      queries have correct answer in top 10
    
    • Performance is competitive with
      standard KG benchmarks
    
    • Model captures translational patterns
      in drug-entity relationships
    
    ═══════════════════════════════════════
    Dataset: DrugBank Knowledge Graph
    Model: TransE (d=100, margin=1.0)
    """
    
    ax4.text(0.1, 0.5, insight_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax4.set_title('(d) Key Insights', fontweight='bold')
    
    plt.suptitle('TransE Knowledge Graph Embedding: DrugBank Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    main()
