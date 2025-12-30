"""
Unify DrugBank and UniProt Knowledge Graphs
============================================
Merges drugbank_facts.txt and uniprot_kg.txt into a single unified KG.

Features:
- Removes duplicate triples
- Provides statistics on entities, relations, and overlap
- Saves unified dataset
"""

import pandas as pd
import os


def load_triples(path: str, source_name: str = None) -> pd.DataFrame:
    """Load triples from a tab-separated file."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )
    if source_name:
        df["origin"] = source_name
    return df


def analyze_dataset(df: pd.DataFrame, name: str):
    """Print statistics about a dataset."""
    entities = set(df["source"]) | set(df["target"])
    relations = set(df["relation"])
    
    print(f"\n{name}:")
    print(f"  Triples:   {len(df):,}")
    print(f"  Entities:  {len(entities):,}")
    print(f"  Relations: {len(relations):,}")
    print(f"  Relations: {sorted(relations)}")
    
    return entities, relations


def unify_datasets(
    drugbank_path: str = "drugbank_facts.txt",
    uniprot_path: str = "uniprot_kg.txt",
    output_path: str = "unified_kg.txt",
    remove_duplicates: bool = True
):
    """
    Unify DrugBank and UniProt knowledge graphs.
    
    Args:
        drugbank_path: Path to DrugBank facts file
        uniprot_path: Path to UniProt KG file
        output_path: Path for unified output
        remove_duplicates: Whether to remove duplicate triples
        
    Returns:
        Unified DataFrame
    """
    print("=" * 60)
    print("Unifying Knowledge Graphs")
    print("=" * 60)
    
    # Load datasets
    print(f"\nLoading {drugbank_path}...")
    drugbank_df = load_triples(drugbank_path, "drugbank")
    
    print(f"Loading {uniprot_path}...")
    uniprot_df = load_triples(uniprot_path, "uniprot")
    
    # Analyze individual datasets
    drugbank_entities, drugbank_relations = analyze_dataset(drugbank_df, "DrugBank")
    uniprot_entities, uniprot_relations = analyze_dataset(uniprot_df, "UniProt")
    
    # Analyze overlap
    shared_entities = drugbank_entities & uniprot_entities
    shared_relations = drugbank_relations & uniprot_relations
    
    print(f"\nOverlap Analysis:")
    print(f"  Shared entities:  {len(shared_entities):,}")
    print(f"  Shared relations: {len(shared_relations)}")
    if shared_entities:
        print(f"  Sample shared entities: {list(shared_entities)[:10]}")
    if shared_relations:
        print(f"  Shared relations: {sorted(shared_relations)}")
    
    # Concatenate datasets
    unified_df = pd.concat([drugbank_df, uniprot_df], ignore_index=True)
    print(f"\nAfter concatenation: {len(unified_df):,} triples")
    
    # Remove duplicates (based on source, relation, target)
    if remove_duplicates:
        before = len(unified_df)
        unified_df = unified_df.drop_duplicates(subset=["source", "relation", "target"])
        after = len(unified_df)
        print(f"Removed {before - after:,} duplicate triples")
    
    # Final statistics
    all_entities = set(unified_df["source"]) | set(unified_df["target"])
    all_relations = set(unified_df["relation"])
    
    print(f"\n{'=' * 60}")
    print("Unified Knowledge Graph Statistics")
    print(f"{'=' * 60}")
    print(f"  Total triples:   {len(unified_df):,}")
    print(f"  Total entities:  {len(all_entities):,}")
    print(f"  Total relations: {len(all_relations):,}")
    print(f"  All relations:   {sorted(all_relations)}")
    
    # Distribution by origin
    print(f"\nTriples by origin:")
    origin_counts = unified_df["origin"].value_counts()
    for origin, count in origin_counts.items():
        print(f"  {origin}: {count:,} ({100*count/len(unified_df):.1f}%)")
    
    # Save unified dataset (without the origin column for training)
    output_df = unified_df[["source", "relation", "target"]]
    output_df.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"\nSaved unified KG to: {output_path}")
    
    # Also save version with origin tracking
    origin_path = output_path.replace(".txt", "_with_origin.txt")
    unified_df.to_csv(origin_path, sep="\t", header=False, index=False)
    print(f"Saved version with origin to: {origin_path}")
    
    return unified_df


if __name__ == "__main__":
    # Paths
    DRUGBANK_PATH = "drugbank_facts.txt"
    UNIPROT_PATH = "uniprot_kg.txt"
    OUTPUT_PATH = "unified_kg.txt"
    
    # Unify datasets
    unified_df = unify_datasets(
        drugbank_path=DRUGBANK_PATH,
        uniprot_path=UNIPROT_PATH,
        output_path=OUTPUT_PATH,
        remove_duplicates=True
    )
