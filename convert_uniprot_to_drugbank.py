"""
Convert UniProt Triples to DrugBank Format
==========================================
This script:
1. Loads the UniProt knowledge graph
2. Extracts TARGET_OF_DRUG triples (Protein, TARGET_OF_DRUG, Drug)
3. Filters to entities that exist in the trained DrugBank model
4. Converts to DrugBank format (Drug, DRUG_TARGET, Protein)
5. Saves the filtered triples for evaluation

Output:
- data/uniprot_filtered.txt: Filtered triples in DrugBank format
"""

import os
import pandas as pd


def load_model_entities(entity2id_path: str) -> set:
    """Load entities from trained model."""
    df = pd.read_csv(entity2id_path)
    return set(df['entity'].values)


def load_uniprot_triples(uniprot_path: str) -> pd.DataFrame:
    """Load UniProt knowledge graph."""
    df = pd.read_csv(
        uniprot_path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )
    return df


def filter_and_convert(
    uniprot_df: pd.DataFrame,
    model_entities: set,
    relation_filter: str = "TARGET_OF_DRUG"
) -> pd.DataFrame:
    """
    Filter UniProt triples and convert to DrugBank format.
    
    UniProt format: (Protein, TARGET_OF_DRUG, Drug)
    DrugBank format: (Drug, DRUG_TARGET, Protein)
    
    Args:
        uniprot_df: Full UniProt knowledge graph
        model_entities: Set of entities in trained model
        relation_filter: Relation type to extract
        
    Returns:
        DataFrame with converted triples
    """
    # Extract TARGET_OF_DRUG triples
    target_triples = uniprot_df[uniprot_df['relation'] == relation_filter].copy()
    
    print(f"Total {relation_filter} triples: {len(target_triples)}")
    
    # In UniProt: source=Protein, target=Drug
    uniprot_proteins = set(target_triples['source'].values)
    uniprot_drugs = set(target_triples['target'].values)
    
    print(f"Unique proteins in UniProt: {len(uniprot_proteins)}")
    print(f"Unique drugs in UniProt: {len(uniprot_drugs)}")
    
    # Find overlaps
    overlapping_proteins = uniprot_proteins & model_entities
    overlapping_drugs = uniprot_drugs & model_entities
    
    print(f"\nOverlapping proteins: {len(overlapping_proteins)} / {len(uniprot_proteins)} "
          f"({100*len(overlapping_proteins)/len(uniprot_proteins):.1f}%)")
    print(f"Overlapping drugs: {len(overlapping_drugs)} / {len(uniprot_drugs)} "
          f"({100*len(overlapping_drugs)/len(uniprot_drugs):.1f}%)")
    
    # Filter triples where BOTH entities exist in model
    filtered = target_triples[
        (target_triples['source'].isin(model_entities)) &
        (target_triples['target'].isin(model_entities))
    ].copy()
    
    print(f"\nFiltered triples (both entities in model): {len(filtered)}")
    
    # Convert to DrugBank format:
    # UniProt: (Protein, TARGET_OF_DRUG, Drug) → DrugBank: (Drug, DRUG_TARGET, Protein)
    converted = pd.DataFrame({
        'source': filtered['target'],      # Drug (was target in UniProt)
        'relation': 'DRUG_TARGET',         # Convert relation name
        'target': filtered['source']       # Protein (was source in UniProt)
    })
    
    # Remove duplicates (same drug-target pair)
    converted = converted.drop_duplicates()
    
    print(f"After removing duplicates: {len(converted)} triples")
    
    return converted


def main():
    # Paths
    UNIPROT_PATH = "uniprot_kg.txt"
    ENTITY2ID_PATH = "outputs/entity2id.csv"
    OUTPUT_DIR = "data"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "uniprot_filtered.txt")
    
    print("=" * 60)
    print("UniProt to DrugBank Format Conversion")
    print("=" * 60)
    
    # Load model entities
    print("\n[1/3] Loading trained model entities...")
    model_entities = load_model_entities(ENTITY2ID_PATH)
    print(f"Total entities in model: {len(model_entities)}")
    
    # Load UniProt
    print("\n[2/3] Loading UniProt knowledge graph...")
    uniprot_df = load_uniprot_triples(UNIPROT_PATH)
    print(f"Total UniProt triples: {len(uniprot_df)}")
    
    # Get relation statistics
    print("\nUniProt relation types:")
    print(uniprot_df['relation'].value_counts().head(10))
    
    # Filter and convert
    print("\n[3/3] Filtering and converting triples...")
    converted_df = filter_and_convert(uniprot_df, model_entities)
    
    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    converted_df.to_csv(OUTPUT_PATH, sep="\t", header=False, index=False)
    
    print("\n" + "=" * 60)
    print(f"Saved {len(converted_df)} triples to: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Show sample output
    print("\nSample converted triples (DrugBank format):")
    print(converted_df.head(10).to_string(index=False))
    
    # Statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Unique drugs: {converted_df['source'].nunique()}")
    print(f"Unique proteins: {converted_df['target'].nunique()}")
    print(f"Total drug-target pairs: {len(converted_df)}")
    
    return converted_df


if __name__ == "__main__":
    main()
