"""
ComplEx Embedding Visualization
================================
Visualize ComplEx embeddings using t-SNE, PCA, and other techniques.
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

OUTPUT_DIR = "outputs_complex"
FIGURES_DIR = "outputs_complex/figures"


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
    # Define color palette for entity types
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
    # Sample if too many entities
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        if entities:
            entities = [entities[i] for i in idx]
        print(f"Sampled {n_samples} entities for visualization")
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         alpha=0.6, s=20, c=range(len(emb_2d)), cmap='viridis')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('ComplEx Entity Embeddings (t-SNE)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig, emb_2d


# ----------------------------
# 1b. Entity Embedding t-SNE (Colored by Type)
# ----------------------------
def plot_entity_tsne_by_type(embeddings: np.ndarray, entities: list,
                              n_samples: int = 2000, perplexity: int = 30,
                              save_path: str = None):
    """
    Create t-SNE visualization colored by entity type.
    """
    # Sample if too many entities
    sampled_entities = entities
    sampled_embeddings = embeddings
    
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        sampled_embeddings = embeddings[idx]
        sampled_entities = [entities[i] for i in idx]
        print(f"Sampled {n_samples} entities for visualization")
    
    # Get colors by entity type
    colors, type_colors, entity_types = get_entity_colors(sampled_entities)
    
    print("Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    emb_2d = tsne.fit_transform(sampled_embeddings)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each entity type separately for legend
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
    ax.set_title('ComplEx Entity Embeddings by Type (t-SNE)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig, emb_2d
    return fig, emb_2d


# ----------------------------
# 2. Entity Embedding PCA
# ----------------------------
def plot_entity_pca(embeddings: np.ndarray, entities: list = None,
                    n_samples: int = 2000, save_path: str = None):
    """
    Create PCA visualization (faster than t-SNE).
    """
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        if entities:
            entities = [entities[i] for i in idx]
    
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         alpha=0.6, s=20, c=range(len(emb_2d)), cmap='plasma')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('ComplEx Entity Embeddings (PCA)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig, emb_2d


# ----------------------------
# 3. Relation Embeddings
# ----------------------------
def plot_relation_embeddings(embeddings: np.ndarray, relations: list,
                              save_path: str = None):
    """
    Visualize relation embeddings with labels.
    """
    # Use PCA for relations
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
    ax.set_title('ComplEx Relation Embeddings (PCA)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 4. Embedding Distribution
# ----------------------------
def plot_embedding_distribution(embeddings: np.ndarray, save_path: str = None):
    """
    Plot distribution of embedding values (real and imaginary parts).
    """
    dim = embeddings.shape[1] // 2  # ComplEx has real + imaginary
    real_part = embeddings[:, :dim]
    imag_part = embeddings[:, dim:]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Real part histogram
    ax1 = axes[0, 0]
    ax1.hist(real_part.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Real Part Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Imaginary part histogram
    ax2 = axes[0, 1]
    ax2.hist(imag_part.flatten(), bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Imaginary Part Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Norm distribution (full embedding)
    ax3 = axes[1, 0]
    norms = np.linalg.norm(embeddings, axis=1)
    ax3.hist(norms, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('L2 Norm', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Full Embedding Norm Distribution', fontsize=12, fontweight='bold')
    ax3.axvline(x=np.mean(norms), color='red', linestyle='--', 
                label=f'Mean: {np.mean(norms):.2f}')
    ax3.legend()
    
    # Complex magnitude per dimension
    ax4 = axes[1, 1]
    complex_mag = np.sqrt(real_part**2 + imag_part**2)
    ax4.hist(complex_mag.flatten(), bins=100, color='purple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Magnitude |z|', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Complex Magnitude Distribution', fontsize=12, fontweight='bold')
    
    plt.suptitle('ComplEx Embedding Value Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 5. Real vs Imaginary Comparison
# ----------------------------
def plot_real_vs_imaginary(embeddings: np.ndarray, n_samples: int = 1000, 
                            save_path: str = None):
    """
    Compare real and imaginary part structure.
    """
    dim = embeddings.shape[1] // 2
    real_part = embeddings[:, :dim]
    imag_part = embeddings[:, dim:]
    
    # Sample entities
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        real_part = real_part[idx]
        imag_part = imag_part[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # t-SNE on real part
    print("Computing t-SNE for real part...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    real_2d = tsne.fit_transform(real_part)
    
    axes[0].scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.5, s=15, c='steelblue')
    axes[0].set_title('Real Part (t-SNE)', fontsize=14, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # t-SNE on imaginary part
    print("Computing t-SNE for imaginary part...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    imag_2d = tsne.fit_transform(imag_part)
    
    axes[1].scatter(imag_2d[:, 0], imag_2d[:, 1], alpha=0.5, s=15, c='coral')
    axes[1].set_title('Imaginary Part (t-SNE)', fontsize=14, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.suptitle('ComplEx: Real vs Imaginary Component Structure', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 6. Nearest Neighbors
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
    
    # Compute distances
    distances = np.linalg.norm(embeddings - query_emb, axis=1)
    nearest_idx = np.argsort(distances)[:k+1]
    
    print(f"\nNearest neighbors for '{query_entity}':")
    print("-" * 50)
    for i, idx in enumerate(nearest_idx):
        print(f"{i+1:2d}. {entities[idx]:40s} (dist: {distances[idx]:.4f})")
    
    return [(entities[idx], distances[idx]) for idx in nearest_idx]


# ----------------------------
# 7. Training Loss Curve
# ----------------------------
def plot_training_loss(loss_csv_path: str, save_path: str = None):
    """
    Plot training loss curve.
    """
    df = pd.read_csv(loss_csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(df['epoch'], df['loss'], color='coral', linewidth=2, label='Training Loss')
    ax.fill_between(df['epoch'], df['loss'], alpha=0.3, color='coral')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('ComplEx Training Loss Curve', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11)
    
    # Mark minimum
    min_idx = df['loss'].idxmin()
    min_epoch = df.loc[min_idx, 'epoch']
    min_loss = df['loss'].min()
    ax.scatter([min_epoch], [min_loss], color='red', s=100, zorder=5)
    ax.annotate(f'Min: {min_loss:.4f}', (min_epoch, min_loss), 
               xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# 8. Evaluation Metrics Bar Chart
# ----------------------------
def plot_metrics(metrics: dict, save_path: str = None):
    """
    Create a bar chart of evaluation metrics.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    bars = ax.bar(metric_names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_title('ComplEx Link Prediction Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dirs()
    
    print("=" * 60)
    print("ComplEx Embedding Visualization")
    print("=" * 60)
    
    # 1. Training loss
    loss_path = os.path.join(OUTPUT_DIR, "training_loss.csv")
    if os.path.exists(loss_path):
        print("\n[1/6] Generating training loss curve...")
        plot_training_loss(loss_path, save_path=os.path.join(FIGURES_DIR, "training_loss.png"))
    
    # 2. Metrics bar chart
    print("\n[2/6] Generating metrics bar chart...")
    metrics = {
        'MRR': 0.5521,
        'Hits@1': 0.4923,
        'Hits@3': 0.5797,
        'Hits@10': 0.6645
    }
    plot_metrics(metrics, save_path=os.path.join(FIGURES_DIR, "metrics_bar_chart.png"))
    
    # 3. Entity embeddings t-SNE
    ent_path = os.path.join(OUTPUT_DIR, "entity_embeddings.csv")
    if os.path.exists(ent_path):
        print("\n[3/7] Generating entity t-SNE visualization...")
        entities, ent_emb = load_embeddings(ent_path)
        print(f"Loaded {len(entities)} entities, dim={ent_emb.shape[1]} (real+imag)")
        
        plot_entity_tsne(ent_emb, entities, n_samples=2000,
                        save_path=os.path.join(FIGURES_DIR, "entity_tsne.png"))
        
        # 3b. Entity embeddings by type
        print("\n[4/7] Generating entity t-SNE colored by type...")
        plot_entity_tsne_by_type(ent_emb, entities, n_samples=2000,
                                 save_path=os.path.join(FIGURES_DIR, "entity_tsne_by_type.png"))
        
        # 5. Embedding distributions
        print("\n[5/7] Generating embedding distribution plots...")
        plot_embedding_distribution(ent_emb, 
                                   save_path=os.path.join(FIGURES_DIR, "embedding_distribution.png"))
        
        # 6. Real vs Imaginary
        print("\n[6/7] Comparing real vs imaginary components...")
        plot_real_vs_imaginary(ent_emb, n_samples=1500,
                              save_path=os.path.join(FIGURES_DIR, "real_vs_imaginary.png"))
    
    # 7. Relation embeddings
    rel_path = os.path.join(OUTPUT_DIR, "relation_embeddings.csv")
    if os.path.exists(rel_path):
        print("\n[7/7] Generating relation embedding plot...")
        relations, rel_emb = load_embeddings(rel_path)
        print(f"Loaded {len(relations)} relations")
        
        plot_relation_embeddings(rel_emb, relations,
                                save_path=os.path.join(FIGURES_DIR, "relation_embeddings.png"))
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
