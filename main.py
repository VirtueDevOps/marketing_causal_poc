# --- start monkey-patch for DoWhy + NetworkX compatibility ---
import networkx as _nx
from networkx.algorithms.d_separation import d_separated as _dsep
_nx.algorithms.d_separated = _dsep
# --- end monkey-patch ---

import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from owlready2 import get_ontology
from dowhy import CausalModel
import argparse

def load_data(data_path):
    if data_path.endswith(".zip"):
        with zipfile.ZipFile(data_path, 'r') as z:
            z.extractall("data")
        data_csv = os.path.join("data", "marketing_data.csv")
    else:
        data_csv = data_path
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} rows from {data_csv}")
    return df

def load_ontology(ont_path):
    onto = get_ontology(ont_path).load()
    print(f"Loaded ontology with classes: {[c.name for c in onto.classes()]}")
    return onto

def run_causal_analysis(df):
    graph = """
    digraph {
        Seasonality -> CampaignSpend;
        Seasonality -> Conversions;
        CampaignSpend   -> Impressions;
        Impressions     -> Clicks;
        Clicks          -> Conversions;
    }
    """
    model = CausalModel(data=df,
                        treatment="CampaignSpend",
                        outcome="Conversions",
                        graph=graph)
    estimand = model.identify_effect()
    print("\nEstimand:", estimand)

    estimate = model.estimate_effect(estimand,
                                     method_name="backdoor.linear_regression")
    print(f"\nEstimated ATE: {estimate.value:.3f} conversions per $1 spend")

    # What-if: +20% spend
    original_mean = df["Conversions"].mean()
    lift = 0.20 * estimate.value
    new_mean = original_mean + lift
    print(f"\nPredicted avg conversions with +20% spend: {new_mean:.3f}")

    # Plot
    plt.scatter(df["CampaignSpend"], df["Conversions"],
                alpha=0.4, label="Observed")
    plt.axhline(original_mean, color="gray", linestyle="--", label="Obs Avg")
    plt.scatter(df["CampaignSpend"]*1.2,
                [new_mean]*len(df),
                alpha=0.4, label="Counterfactual")
    plt.xlabel("Campaign Spend")
    plt.ylabel("Conversions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("causal_results.png")
    print("Plot saved to causal_results.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to .zip or .csv")
    p.add_argument("--ontology", required=True, help="Path to .owl")
    args = p.parse_args()

    df = load_data(args.data)
    _  = load_ontology(args.ontology)
    run_causal_analysis(df)
