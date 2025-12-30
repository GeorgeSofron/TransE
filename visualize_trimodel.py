"""
TriModel Embedding Visualization
=================================
Visualize TriModel embeddings using t-SNE, PCA, and other techniques.

TriModel has 3 embedding vectors per entity (v1, v2, v3), which are 
concatenated in the CSV. This script visualizes them individually and combined.
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

OUTPUT_DIR = "outputs_trimodel"
FIGURES_DIR = "outputs_trimodel/figures"


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


def split_trimodel_embeddings(embeddings: np.ndarray) -> tuple:
    """
    Split TriModel embeddings into their 3 components.
    TriModel stores [v1 | v2 | v3] concatenated.
    
    Returns:
        (v1, v2, v3) each of shape (n_entities, dim)
    """
    dim = embeddings.shape[1] // 3
    v1 = embeddings[:, :dim]
    v2 = embeddings[:, dim:2*dim]
    v3 = embeddings[:, 2*dim:]
    return v1, v2, v3


# ----------------------------
# 1. Entity Embedding t-SNE (Full)
# ----------------------------
def plot_entity_tsne(embeddings: np.ndarray, entities: list = None,
                     n_samples: int = 2000, perplexity: int = 30,
                     save_path: str = None):
    """
    Create t-SNE visualization of full entity embeddings (all 3 components).
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
    ax.set_title('TriModel Entity Embeddings (t-SNE)', fontsize=14, fontweight='bold')
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
    ax.set_title('TriModel Entity Embeddings by Type (t-SNE)', fontsize=14, fontweight='bold')
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
# 3. Compare Three Components
# ----------------------------
def plot_three_components(embeddings: np.ndarray, entities: list,
                          n_samples: int = 1000, save_path: str = None):
    """
    Compare t-SNE visualizations of all three embedding components (v1, v2, v3).
    """
    v1, v2, v3 = split_trimodel_embeddings(embeddings)
    
    sampled_entities = entities
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        v1 = v1[idx]
        v2 = v2[idx]
        v3 = v3[idx]
        sampled_entities = [entities[i] for i in idx]
        print(f"Sampled {n_samples} entities for visualization")
    
    colors, type_colors, entity_types = get_entity_colors(sampled_entities)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    components = [('v1 (Head)', v1), ('v2 (Relation)', v2), ('v3 (Tail)', v3)]
    
    for ax, (name, comp) in zip(axes, components):
        print(f"Computing t-SNE for {name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        emb_2d = tsne.fit_transform(comp)
        
        for entity_type, color in type_colors.items():
            mask = [t == entity_type for t in entity_types]
            if any(mask):
                points = emb_2d[mask]
                ax.scatter(points[:, 0], points[:, 1], 
                          alpha=0.6, s=20, c=color, label=entity_type,
                          edgecolors='white', linewidth=0.2)
        
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('TriModel: Three Embedding Components (t-SNE)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


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
    ax.set_title('TriModel Relation Embeddings (PCA)', fontsize=14, fontweight='bold')
    
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
    Plot distribution of embedding values for all three components.
    """
    v1, v2, v3 = split_trimodel_embeddings(embeddings)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # v1 histogram
    ax1 = axes[0, 0]
    ax1.hist(v1.flatten(), bins=100, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('v1 Component Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # v2 histogram
    ax2 = axes[0, 1]
    ax2.hist(v2.flatten(), bins=100, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('v2 Component Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # v3 histogram
    ax3 = axes[1, 0]
    ax3.hist(v3.flatten(), bins=100, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('v3 Component Distribution', fontsize=12, fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Norm distribution (full embedding)
    ax4 = axes[1, 1]
    norms = np.linalg.norm(embeddings, axis=1)
    ax4.hist(norms, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('L2 Norm', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Full Embedding Norm Distribution', fontsize=12, fontweight='bold')
    ax4.axvline(x=np.mean(norms), color='red', linestyle='--', 
                label=f'Mean: {np.mean(norms):.2f}')
    ax4.legend()
    
    plt.suptitle('TriModel Embedding Value Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


# ----------------------------
# 6. Component Correlation
# ----------------------------
def plot_component_correlation(embeddings: np.ndarray, n_samples: int = 1000,
                                save_path: str = None):
    """
    Analyze correlation between the three embedding components.
    """
    v1, v2, v3 = split_trimodel_embeddings(embeddings)
    
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        v1 = v1[idx]
        v2 = v2[idx]
        v3 = v3[idx]
    
    # Compute norms
    v1_norms = np.linalg.norm(v1, axis=1)
    v2_norms = np.linalg.norm(v2, axis=1)
    v3_norms = np.linalg.norm(v3, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # v1 vs v2
    axes[0].scatter(v1_norms, v2_norms, alpha=0.4, s=10, c='#3498db')
    corr12 = np.corrcoef(v1_norms, v2_norms)[0, 1]
    axes[0].set_xlabel('v1 Norm', fontsize=11)
    axes[0].set_ylabel('v2 Norm', fontsize=11)
    axes[0].set_title(f'v1 vs v2 (r={corr12:.3f})', fontsize=12, fontweight='bold')
    
    # v1 vs v3
    axes[1].scatter(v1_norms, v3_norms, alpha=0.4, s=10, c='#e74c3c')
    corr13 = np.corrcoef(v1_norms, v3_norms)[0, 1]
    axes[1].set_xlabel('v1 Norm', fontsize=11)
    axes[1].set_ylabel('v3 Norm', fontsize=11)
    axes[1].set_title(f'v1 vs v3 (r={corr13:.3f})', fontsize=12, fontweight='bold')
    
    # v2 vs v3
    axes[2].scatter(v2_norms, v3_norms, alpha=0.4, s=10, c='#2ecc71')
    corr23 = np.corrcoef(v2_norms, v3_norms)[0, 1]
    axes[2].set_xlabel('v2 Norm', fontsize=11)
    axes[2].set_ylabel('v3 Norm', fontsize=11)
    axes[2].set_title(f'v2 vs v3 (r={corr23:.3f})', fontsize=12, fontweight='bold')
    
    plt.suptitle('TriModel: Component Norm Correlations', fontsize=14, fontweight='bold')
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
    ax.set_title('TriModel Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add min loss annotation
    min_loss = df['loss'].min()
    min_epoch = df.loc[df['loss'].idxmin(), 'epoch']
    ax.annotate(f'Min: {min_loss:.4f}', xy=(min_epoch, min_loss),
               xytext=(min_epoch + 5, min_loss + 0.1),
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
                    key, val = line.strip().split(':')
                    try:
                        metrics[key.strip()] = float(val.strip())
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
    ax.set_title('TriModel Evaluation Metrics', fontsize=14, fontweight='bold')
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
    
    dim = entity_emb.shape[1] // 3
    print(f"Loaded {len(entities)} entities with dimension {entity_emb.shape[1]} (3 x {dim})")
    print(f"Loaded {len(relations)} relations with dimension {relation_emb.shape[1]} (3 x {dim})")
    
    # 1. Entity t-SNE (full embeddings)
    print("\n1. Creating entity t-SNE visualization...")
    plot_entity_tsne(entity_emb, entities, n_samples=2000,
                     save_path=f"{FIGURES_DIR}/entity_tsne.png")
    
    # 2. Entity t-SNE by type
    print("\n2. Creating entity t-SNE by type...")
    plot_entity_tsne_by_type(entity_emb, entities, n_samples=2000,
                              save_path=f"{FIGURES_DIR}/entity_tsne_by_type.png")
    
    # 3. Three components comparison
    print("\n3. Creating three components comparison...")
    plot_three_components(entity_emb, entities, n_samples=1000,
                          save_path=f"{FIGURES_DIR}/three_components.png")
    
    # 4. Relation embeddings
    print("\n4. Creating relation embeddings visualization...")
    plot_relation_embeddings(relation_emb, relations,
                              save_path=f"{FIGURES_DIR}/relation_embeddings.png")
    
    # 5. Embedding distribution
    print("\n5. Creating embedding distribution plot...")
    plot_embedding_distribution(entity_emb,
                                 save_path=f"{FIGURES_DIR}/embedding_distribution.png")
    
    # 6. Component correlation
    print("\n6. Creating component correlation plot...")
    plot_component_correlation(entity_emb, n_samples=2000,
                                save_path=f"{FIGURES_DIR}/component_correlation.png")
    
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
