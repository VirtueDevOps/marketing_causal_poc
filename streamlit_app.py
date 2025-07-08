# -----------------------------------------------------------------------
# Stub out all dowhy.refuters (so no numpy.distutils or sklearn is ever needed)
# -----------------------------------------------------------------------
import sys, types

# 1) Stub numpy.distutils.misc_util to avoid the missing numpy.distutils error
_misc = types.SimpleNamespace(is_sequence=lambda x: False)
sys.modules["numpy.distutils.misc_util"] = _misc

# 2) Stub econml estimator so import dowhy.causal_estimators.econml never fails
_ec = types.ModuleType("dowhy.causal_estimators.econml")
_ec.Econml = type("Econml", (), {})  # no-op
sys.modules["dowhy.causal_estimators.econml"] = _ec

# 3) Stub all the refuter modules & their key classes/functions
_ref_base = types.ModuleType("dowhy.causal_refuters")
sys.modules["dowhy.causal_refuters"] = _ref_base

for sub, clsname in [
    ("add_unobserved_common_cause", "AddUnobservedCommonCause"),
    ("graph_refuter",            "GraphRefuter"),
    ("placebo_treatment_refuter","PlaceboTreatmentRefuter"),
    ("data_subset_refuter",      "DataSubsetRefuter"),
    ("random_common_cause",      "RandomCommonCauseRefuter")
]:
    mod = types.ModuleType(f"dowhy.causal_refuters.{sub}")
    setattr(mod, clsname, type(clsname, (), {}))
    sys.modules[f"dowhy.causal_refuters.{sub}"] = mod


# -----------------------------------------------------------------------
# Now import everything else
# -----------------------------------------------------------------------
import os, io, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import networkx as _nx
from owlready2 import get_ontology
from dowhy import CausalModel
from networkx.algorithms.d_separation import d_separated as _dsep

# Monkey-patch NetworkX for DoWhy compatibility
_nx.algorithms.d_separated = _dsep

st.set_page_config(layout="wide")

# 📖 Tutorial toggle
if st.sidebar.checkbox("📖 Show Tutorial"):
    with open("TUTORIAL.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())
    st.stop()

st.title("📈 Ontology + Causal AI Demo")

# — Sidebar: Controls & Reset —
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset App"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# 1) Data
st.sidebar.subheader("1) Data")
mode = st.sidebar.radio("Source", ["Synthetic", "Upload"])
if mode == "Synthetic":
    n    = st.sidebar.slider("Samples", 100, 5000, 500)
    seed = st.sidebar.number_input("Seed", 0, 9999, 0)
    np.random.seed(seed)

    df = pd.DataFrame({"CampaignSpend": np.random.uniform(100, 1000, n)})
    df["Seasonality"]  = np.random.choice([0,1], size=n)
    df["CampaignSpend"] += df["Seasonality"] * 200

    df["AdQuality"]     = np.random.choice(["Low","High"], size=n)
    df["CampaignSpend"] += (df["AdQuality"]=="High") * 0.15 * df["CampaignSpend"]

    df["ChannelType"]   = np.random.choice(["Search","Social","Display"], size=n)
    df["CampaignSpend"] *= np.where(df["ChannelType"]=="Social", 0.90, 1.0)

    df["CustomerSegment"] = np.random.choice(["A","B"], size=n)
    seg_eff = np.where(df["CustomerSegment"]=="A", 0.05, 0.02)

    df["Impressions"] = df["CampaignSpend"]*10 + np.random.normal(0,100,n)
    df["Clicks"]      = df["Impressions"]*0.05 + np.random.normal(0,5,n)
    df["Conversions"] = (
        df["Clicks"]*0.1
        + np.random.normal(0,2,n)
        + df["Seasonality"]*5
        + df["CampaignSpend"]*seg_eff
        + (df["AdQuality"]=="High")*3
        - (df["ChannelType"]=="Display")*2
    )
else:
    uploaded = st.sidebar.file_uploader("ZIP/CSV", ["zip","csv"])
    if not uploaded:
        st.stop()
    if uploaded.name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(uploaded.read())) as z:
            for f in z.namelist():
                if f.endswith(".csv"):
                    df = pd.read_csv(z.open(f))
    else:
        df = pd.read_csv(uploaded)

st.write("### Data sample", df.head())

# 2) Ontology
st.sidebar.subheader("2) Ontology")
ont_mode = st.sidebar.radio("Source", ["Built-In", "Upload"])
if ont_mode=="Built-In":
    onto = get_ontology("ontology/marketing_ontology.owl").load()
else:
    upl = st.sidebar.file_uploader("OWL file", ["owl"])
    if not upl:
        st.stop()
    with open("temp.owl","wb") as f:
        f.write(upl.getvalue())
    onto = get_ontology(f"file://{os.path.abspath('temp.owl')}").load()

st.write("Ontology classes", [c.name for c in onto.classes()])

# 3) Run Causal Model
st.sidebar.subheader("3) Run Model")
if st.sidebar.button("▶️ Run Causal Model"):
    graph = """
    digraph {
        Seasonality       -> CampaignSpend;
        AdQuality         -> CampaignSpend;
        ChannelType       -> CampaignSpend;
        CustomerSegment   -> CampaignSpend;
        Seasonality       -> Conversions;
        AdQuality         -> Conversions;
        ChannelType       -> Conversions;
        CustomerSegment   -> Conversions;
        CampaignSpend     -> Impressions;
        Impressions       -> Clicks;
        Clicks            -> Conversions;
    }
    """
    model    = CausalModel(df, "CampaignSpend", "Conversions", graph)
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

    st.session_state.update({
        "ate":       estimate.value,
        "base_mean": df["Conversions"].mean(),
        "df":        df,
        "graph":     graph
    })

# 4) Results & What-If
if "ate" in st.session_state:
    ate       = st.session_state["ate"]
    base_mean = st.session_state["base_mean"]
    df        = st.session_state["df"]
    graph     = st.session_state["graph"]

    st.write("## 📊 Causal Results")
    st.subheader("Causal Graph")
    st.graphviz_chart(graph)

    st.write(f"**Overall ATE:** {ate:.4f} conversions per $1 spend")

    st.write("### Heterogeneous Effects by Customer Segment")
    rows = []
    for seg in sorted(df["CustomerSegment"].unique()):
        dseg = df[df["CustomerSegment"]==seg]
        mseg = CausalModel(dseg, "CampaignSpend", "Conversions", graph)
        est  = mseg.estimate_effect(mseg.identify_effect(), method_name="backdoor.linear_regression")
        rows.append((seg, est.value))
    st.table(pd.DataFrame(rows, columns=["Segment","ATE"]).round(4))

    st.sidebar.subheader("4) What-If Scenario")
    mult     = st.sidebar.slider("Spend ×", 0.5, 2.0, 1.2, 0.05)
    new_mean = base_mean + (mult - 1) * ate
    st.write(f"**Predicted avg conversions @×{mult:.2f}:** {new_mean:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(df["CampaignSpend"], df["Conversions"], alpha=0.4, label="Observed")
    ax.axhline(base_mean, linestyle="--", color="gray", label="Observed Avg")
    ax.scatter(df["CampaignSpend"]*mult, [new_mean]*len(df),
               alpha=0.4, label="Counterfactual")
    ax.set_xlabel("Campaign Spend (USD)")
    ax.set_ylabel("Conversions")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("▶ Click ‘Run Causal Model’ to compute ATE & enable the slider.")
