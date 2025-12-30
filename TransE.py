import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.trackers import ResultTracker


# -------------------------------------------------
# Load & preprocess KG
# -------------------------------------------------
def load_and_preprocess_kg(file_path, test_size=0.2, random_state=42):
    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"]
    )

    print("Dataset preview:")
    print(data.head())
    print(f"\nTotal triples: {len(data)}")

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Train triples: {len(train_df)}")
    print(f"Test triples:  {len(test_df)}")

    return train_df, test_df


# -------------------------------------------------
# Create TriplesFactory
# -------------------------------------------------
def create_triples_factory(df):
    return TriplesFactory.from_labeled_triples(
        df[["source", "relation", "target"]].values
    )


# -------------------------------------------------
# Train TransE (NO evaluation metrics)
# -------------------------------------------------
def train_transe(training_tf, testing_tf, epochs=50):
    tracker = ResultTracker()

    results = pipeline(
        model="TransE",
        training=training_tf,
        testing=testing_tf,        # required by PyKEEN
        training_loop="sLCWA",
        epochs=epochs,
        random_seed=42,
        device="cpu",              # change to "cuda" if available
        evaluator=None,            # do not use metrics
        result_tracker=tracker,

        model_kwargs=dict(
            embedding_dim=100,
            scoring_fct_norm=1,
        ),

        optimizer_kwargs=dict(
            lr=1e-3,
        ),
    )

    return results, tracker


# -------------------------------------------------
# Save loss curve & embeddings
# -------------------------------------------------
def save_training_outputs(results, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    model = results.model

    # ---- Loss per epoch (from pipeline results) ----
    losses = results.losses

    loss_df = pd.DataFrame({
        "epoch": range(1, len(losses) + 1),
        "loss": losses
    })
    loss_df.to_csv(os.path.join(output_dir, "training_loss.csv"), index=False)

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TransE Training Loss")
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    # ---- Entity embeddings ----
    entity_emb = model.entity_representations[0]().detach().cpu().numpy()
    triples_factory = results.training
    id_to_entity = {v: k for k, v in triples_factory.entity_to_id.items()}

    pd.DataFrame(
        entity_emb,
        index=[id_to_entity[i] for i in range(len(entity_emb))]
    ).to_csv(os.path.join(output_dir, "entity_embeddings.csv"))

    # ---- Relation embeddings ----
    relation_emb = model.relation_representations[0]().detach().cpu().numpy()
    id_to_relation = {v: k for k, v in triples_factory.relation_to_id.items()}

    pd.DataFrame(
        relation_emb,
        index=[id_to_relation[i] for i in range(len(relation_emb))]
    ).to_csv(os.path.join(output_dir, "relation_embeddings.csv"))

    print("\nSaved outputs to 'outputs/' directory.")



# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "drugbank_facts.txt"
    OUTPUT_DIR = "outputs"

    train_df, test_df = load_and_preprocess_kg(DATA_PATH)

    training_tf = create_triples_factory(train_df)
    testing_tf  = create_triples_factory(test_df)

    results, tracker = train_transe(
        training_tf,
        testing_tf,
        epochs=100
    )

    save_training_outputs(results, output_dir=OUTPUT_DIR)
