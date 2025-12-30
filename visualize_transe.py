"""
TransE Embedding Visualization
===============================
Visualize TransE embeddings using t-SNE, PCA, and other techniques.

TransE uses a single embedding vector per entity, making visualization 
straightforward compared to ComplEx or TriModel.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = "outputs_transe"
FIGURES_DIR = "outputs_transe/figures"


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def classify_entity(entity_name: str) -> str:
    """
    Classify entity by its naming pattern in DrugBank.
    """
    if entity_name.startswith('DB'):
        return 'Drug'
    elif entity_name.startswith('P') and entity_name[1:].isdigit():
        return 'Protein/Target'
    elif entity_name.startswith('Q') and len(entity_name) >= 5:
        return 'Protein/Target'
    elif entity_name.startswith('SMP'):
        return 'Pathway'
    elif entity_name.startswith('D') and entity_name[1:].isdigit():
        return 'Category'
    elif entity_name.startswith('ATC:'):
        return 'ATC Code'
    elif entity_name.startswith('BE'):
        return 'Bioentity'
    else:
        return 'Other'


def get_entity_colors(entities: list) -> tuple:
    """
    Get colors for entities based on their type.
    Returns (colors_list, type_to_color_dict, entity_types_list)
    """
    type_colors = {
        'Drug': '#e74c3c',           # Red
        'Protein/Target': '#3498db', # Blue
        'Pathway': '#2ecc71',        # Green
        'Category': '#9b59b6',       # Purple
        'ATC Code': '#f39c12',       # Orange
        'Bioentity': '#1abc9c',      # Teal
        'Other': '#95a5a6'           # Gray
    }
    
    entity_types = [classify_entity(e) for e in entities]
    colors = [type_colors[t] for t in entity_types]
    
    return colors, type_colors, entity_types


def load_embeddings(csv_path: str) -> tuple:
    """Load embeddings from CSV file."""
    df = pd.read_csv(csv_path, index_col=0)
    entities = df.index.tolist()
    embeddings = df.values
    return entities, embeddings


# ----------------------------
# 1. Entity Embedding t-SNE
# ----------------------------
def plot_entity_tsne(embeddings: np.ndarray, entities: list = None,
                     n_samples: int = 2000, perplexity: int = 30,
                     save_path: str = None):
    """
    Create t-SNE visualization of entity embeddings.
    """
    sampled_emb = embeddings
    sampled_ent = entities
    
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_emb = embeddings[idx]
        if entities:
            sampled_ent = [entities[i] for i in idx]
        print(f"Sampled {n_samples} entities for visualization")
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(sampled_emb)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         alpha=0.6, s=20, c=range(len(emb_2d)), cmap='viridis')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('TransE Entity Embeddings (t-SNE)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig, emb_2d


# ----------------------------
# 2. Entity Embedding t-SNE (By Type)
# ----------------------------
def plot_entity_tsne_by_type(embeddings: np.ndarray, entities: list,
                              n_samples: int = 2000, perplexity: int = 30,
                              save_path: str = None):
    """
    Create t-SNE visualization colored by entity type.
    """
    sampled_entities = entities
    sampled_embeddings = embeddings
    
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_embeddings = embeddings[idx]
        sampled_entities = [entities[i] for i in idx]
        print(f"Sampled {n_samples} entities for visualization")
    
    colors, type_colors, entity_types = get_entity_colors(sampled_entities)
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(sampled_embeddings)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for entity_type, color in type_colors.items():
        mask = [t == entity_type for t in entity_types]
        if any(mask):
            points = emb_2d[mask]
            count = sum(mask)
            ax.scatter(points[:, 0], points[:, 1], 
                      alpha=0.6, s=25, c=color, label=f'{entity_type} ({count})',
                      edgecolors='white', linewidth=0.3)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('TransE Entity Embeddings by Type (t-SNE)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig, emb_2d


# ----------------------------
# 3. Entity Embedding PCA
# ----------------------------
def plot_entity_pca(embeddings: np.ndarray, entities: list = None,
                    n_samples: int = 2000, save_path: str = None):
    """
    Create PCA visualization (faster than t-SNE).
    """
    sampled_emb = embeddings
    sampled_ent = entities
    
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_emb = embeddings[idx]
        if entities:
            sampled_ent = [entities[i] for i in idx]
    
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(sampled_emb)
    
    # Get colors
    colors, type_colors, entity_types = get_entity_colors(sampled_ent)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for entity_type, color in type_colors.items():
        mask = [t == entity_type for t in entity_types]
        if any(mask):
            points = emb_2d[mask]
            count = sum(mask)
            ax.scatter(points[:, 0], points[:, 1], 
                      alpha=0.6, s=25, c=color, label=f'{entity_type} ({count})',
                      edgecolors='white', linewidth=0.3)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('TransE Entity Embeddings (PCA)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig, emb_2d


# ----------------------------
# 4. Relation Embeddings
# ----------------------------
def plot_relation_embeddings(embeddings: np.ndarray, relations: list,
                              save_path: str = None):
    """
    Visualize relation embeddings with labels using PCA.
    """
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(relations), 20)))
    
    for i, (x, y) in enumerate(emb_2d):
        color_idx = i % 20
        ax.scatter(x, y, c=[colors[color_idx]], s=200, edgecolors='black', linewidth=1.5)
        ax.annotate(relations[i], (x, y), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title('TransE Relation Embeddings (PCA)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


# ----------------------------
# 5. Embedding Distribution
# ----------------------------
def plot_embedding_distribution(embeddings: np.ndarray, save_path: str = None):
    """
    Plot distribution of embedding values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Full embedding value distribution
    ax1 = axes[0, 0]
    ax1.hist(embeddings.flatten(), bins=100, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Embedding Value Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Per-dimension mean
    ax2 = axes[0, 1]
    dim_means = embeddings.mean(axis=0)
    ax2.bar(range(len(dim_means)), dim_means, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Dimension', fontsize=11)
    ax2.set_ylabel('Mean Value', fontsize=11)
    ax2.set_title('Mean Value per Dimension', fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Embedding norm distribution
    ax3 = axes[1, 0]
    norms = np.linalg.norm(embeddings, axis=1)
    ax3.hist(norms, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('L2 Norm', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Entity Embedding Norm Distribution', fontsize=12, fontweight='bold')
    ax3.axvline(x=np.mean(norms), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(norms):.2f}')
    ax3.legend()
    
    # Per-dimension std
    ax4 = axes[1, 1]
    dim_stds = embeddings.std(axis=0)
    ax4.bar(range(len(dim_stds)), dim_stds, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Dimension', fontsize=11)
    ax4.set_ylabel('Std Dev', fontsize=11)
    ax4.set_title('Standard Deviation per Dimension', fontsize=12, fontweight='bold')
    
    plt.suptitle('TransE Embedding Value Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


# ----------------------------
# 6. Dimension Analysis
# ----------------------------
def plot_dimension_analysis(embeddings: np.ndarray, n_samples: int = 1000,
                             save_path: str = None):
    """
    Analyze which dimensions are most informative.
    """
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
    
    # Compute variance per dimension
    dim_variance = np.var(embeddings, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sorted variance
    ax1 = axes[0]
    sorted_idx = np.argsort(dim_variance)[::-1]
    sorted_var = dim_variance[sorted_idx]
    ax1.bar(range(len(sorted_var)), sorted_var, color='#3498db', alpha=0.7)
    ax1.set_xlabel('Dimension (sorted by variance)', fontsize=11)
    ax1.set_ylabel('Variance', fontsize=11)
    ax1.set_title('Dimension Variance (Sorted)', fontsize=12, fontweight='bold')
    
    # Cumulative explained variance
    ax2 = axes[1]
    cumsum = np.cumsum(sorted_var) / np.sum(sorted_var)
    ax2.plot(range(len(cumsum)), cumsum, 'b-', linewidth=2)
    ax2.axhline(y=0.9, color='red', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    
    # Find how many dimensions for 90%/95%
    n_90 = np.argmax(cumsum >= 0.9) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    ax2.axvline(x=n_90, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Number of Dimensions', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Ratio', fontsize=11)
    ax2.set_title(f'Cumulative Variance (90%: {n_90}d, 95%: {n_95}d)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.suptitle('TransE Dimension Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


# ----------------------------
# 7. Nearest Neighbors
# ----------------------------
def find_nearest_neighbors(embeddings: np.ndarray, entities: list, 
                           query_entity: str, k: int = 10):
    """
    Find k nearest neighbors for a query entity.
    """
    if query_entity not in entities:
        print(f"Entity '{query_entity}' not found!")
        return None
    
    query_idx = entities.index(query_entity)
    query_emb = embeddings[query_idx]
    
    distances = np.linalg.norm(embeddings - query_emb, axis=1)
    nearest_idx = np.argsort(distances)[:k+1]
    
    print(f"\nNearest neighbors for '{query_entity}':")
    print("-" * 50)
    for i, idx in enumerate(nearest_idx):
        etype = classify_entity(entities[idx])
        print(f"{i+1:2d}. {entities[idx]:40s} [{etype:15s}] (dist: {distances[idx]:.4f})")
    
    return [(entities[idx], distances[idx]) for idx in nearest_idx]


# ----------------------------
# 8. Training Loss Curve
# ----------------------------
def plot_training_loss(loss_csv_path: str, save_path: str = None):
    """
    Plot training loss curve.
    """
    df = pd.read_csv(loss_csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('TransE Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add min loss annotation
    min_loss = df['loss'].min()
    min_epoch = df.loc[df['loss'].idxmin(), 'epoch']
    ax.annotate(f'Min: {min_loss:.4f}', xy=(min_epoch, min_loss),
               xytext=(min_epoch + 5, min_loss + 0.05),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


# ----------------------------
# 9. Evaluation Metrics Bar Chart
# ----------------------------
def plot_metrics(metrics_path: str = None, metrics: dict = None, 
                 save_path: str = None):
    """
    Plot evaluation metrics as a bar chart.
    """
    if metrics is None and metrics_path:
        metrics = {}
        with open(metrics_path, 'r') as f:
            for line in f:
                if ':' in line and not line.startswith('='):
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        val = parts[-1].strip()  # Take last part in case of multiple colons
                        try:
                            metrics[key] = float(val)
                        except ValueError:
                            pass
    
    if not metrics:
        print("No metrics to plot!")
        return None
    
    # Select main metrics
    main_metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    values = [metrics.get(m, 0) for m in main_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(main_metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('TransE Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ensure_dirs()
    
    print("Loading embeddings...")
    entities, entity_emb = load_embeddings(f"{OUTPUT_DIR}/entity_embeddings.csv")
    relations, relation_emb = load_embeddings(f"{OUTPUT_DIR}/relation_embeddings.csv")
    
    dim = entity_emb.shape[1]
    print(f"Loaded {len(entities)} entities with dimension {dim}")
    print(f"Loaded {len(relations)} relations with dimension {dim}")
    
    # 1. Entity t-SNE (full embeddings)
    print("\n1. Creating entity t-SNE visualization...")
    plot_entity_tsne(entity_emb, entities, n_samples=2000,
                     save_path=f"{FIGURES_DIR}/entity_tsne.png")
    
    # 2. Entity t-SNE by type
    print("\n2. Creating entity t-SNE by type...")
    plot_entity_tsne_by_type(entity_emb, entities, n_samples=2000,
                              save_path=f"{FIGURES_DIR}/entity_tsne_by_type.png")
    
    # 3. Entity PCA
    print("\n3. Creating entity PCA visualization...")
    plot_entity_pca(entity_emb, entities, n_samples=2000,
                    save_path=f"{FIGURES_DIR}/entity_pca.png")
    
    # 4. Relation embeddings
    print("\n4. Creating relation embeddings visualization...")
    plot_relation_embeddings(relation_emb, relations,
                              save_path=f"{FIGURES_DIR}/relation_embeddings.png")
    
    # 5. Embedding distribution
    print("\n5. Creating embedding distribution plot...")
    plot_embedding_distribution(entity_emb,
                                 save_path=f"{FIGURES_DIR}/embedding_distribution.png")
    
    # 6. Dimension analysis
    print("\n6. Creating dimension analysis plot...")
    plot_dimension_analysis(entity_emb, n_samples=2000,
                             save_path=f"{FIGURES_DIR}/dimension_analysis.png")
    
    # 7. Training loss
    print("\n7. Creating training loss plot...")
    try:
        plot_training_loss(f"{OUTPUT_DIR}/training_loss.csv",
                          save_path=f"{FIGURES_DIR}/training_loss.png")
    except FileNotFoundError:
        print("Training loss file not found, skipping...")
    
    # 8. Metrics bar chart
    print("\n8. Creating metrics bar chart...")
    try:
        plot_metrics(metrics_path=f"{OUTPUT_DIR}/Evaluation.txt",
                    save_path=f"{FIGURES_DIR}/metrics_bar_chart.png")
    except FileNotFoundError:
        print("Evaluation file not found, skipping...")
    
    # 9. Find some nearest neighbors
    print("\n9. Finding nearest neighbors for sample entities...")
    sample_drugs = [e for e in entities if e.startswith('DB')][:3]
    for drug in sample_drugs:
        find_nearest_neighbors(entity_emb, entities, drug, k=5)
    
    print(f"\n{'='*50}")
    print(f"All visualizations saved to {FIGURES_DIR}/")
    print(f"{'='*50}")
