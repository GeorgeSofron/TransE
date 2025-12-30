import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# -------------------------------------------------
# Configuration
# -------------------------------------------------
DATA_PATH = "drugbank_facts.txt"
N_TRIPLES = 200        # choose between 100–300
RANDOM_SEED = 42


# -------------------------------------------------
# Load knowledge graph
# -------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["source", "relation", "target"]
)

print(f"Total triples in KG: {len(df)}")

# Sample a small subgraph
sub_df = df.sample(n=N_TRIPLES, random_state=RANDOM_SEED)
print(f"Sampled triples: {len(sub_df)}")


# -------------------------------------------------
# Build graph
# -------------------------------------------------
G = nx.MultiDiGraph()

for _, row in sub_df.iterrows():
    G.add_edge(
        row["source"],
        row["target"],
        relation=row["relation"]
    )

print(f"Nodes in subgraph: {G.number_of_nodes()}")
print(f"Edges in subgraph: {G.number_of_edges()}")


# -------------------------------------------------
# Compute layout
# -------------------------------------------------
plt.figure(figsize=(12, 12))

# Spring layout works best for small KGs
pos = nx.spring_layout(
    G,
    seed=RANDOM_SEED,
    k=0.8
)


# -------------------------------------------------
# Draw graph
# -------------------------------------------------

# Draw nodes and edges
nx.draw(
    G,
    pos,
    node_size=300,
    node_color="lightblue",
    edge_color="gray",
    with_labels=True,
    font_size=6,
    arrows=True,
    alpha=0.9
)

# Draw edge labels (relationship names)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=6,
    label_pos=0.5
)

plt.title(f"Sample DrugBank Knowledge Graph ({N_TRIPLES} Triples)")
plt.tight_layout()
plt.show()
