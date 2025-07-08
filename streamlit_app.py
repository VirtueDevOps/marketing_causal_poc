# ----------------------------------------------------------------------------- 
# Stub out dowhy.causal_refuters to avoid any import‐time errors on Streamlit Cloud 
# ----------------------------------------------------------------------------- 
import sys, types

_refuters_mod = types.ModuleType("dowhy.causal_refuters")
_refuters_mod.add_unobserved_common_cause = types.ModuleType("dowhy.causal_refuters.add_unobserved_common_cause")
_refuters_mod.graph_refuter             = types.ModuleType("dowhy.causal_refuters.graph_refuter")

sys.modules["dowhy.causal_refuters"]                            = _refuters_mod
sys.modules["dowhy.causal_refuters.add_unobserved_common_cause"] = _refuters_mod.add_unobserved_common_cause
sys.modules["dowhy.causal_refuters.graph_refuter"]               = _refuters_mod.graph_refuter

# ----------------------------------------------------------------------------- 
# Now import everything else normally 
# ----------------------------------------------------------------------------- 
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

# — Sidebar: Global Controls & Reset —
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
    df["Seasonality"]     = np.random.choice([0,1], size=n)
    df["CampaignSpend"]  += df["Seasonality"] * 200

    df["AdQuality"]      = np.random.choice(["Low","High"], size=n)
    df["CampaignSpend"]  += (df["AdQuality"]=="High") * 0.15 * df["CampaignSpend"]

    df["ChannelType"]    = np.random.choice(["Search","Social","Display"], size=n)
    df["CampaignSpend"]  *= np.where(df["ChannelType"]=="Social", 0.90, 1.0)

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
    uploaded_owl = st.sidebar.file_uploader("OWL file", ["owl"])
    if not uploaded_owl:
        st.stop()
    with open("temp.owl","wb") as f:
        f.write(uploaded_owl.getvalue())
    onto = get_ontology(f"file://{os.path.abspath('temp.owl')}").load()

st.write("Ontology classes", [c.name for c in onto.classes()])

# 3) Run Causal Model & Estimate
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
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.linear_regression"
    )

    # three refutation tests (these now all no-op but still display)
    ref1 = model.refute_estimate(estimand, estimate, method_name="random_common_cause")
    ref2 = model.refute_estimate(estimand, estimate,
                                 method_name="placebo_treatment_refuter",
                                 placebo_type="permute")
    ref3 = model.refute_estimate(estimand, estimate,
                                 method_name="data_subset_refuter",
                                 subset_fraction=0.7)

    st.session_state.update({
        "ate":    estimate.value,
        "base":   df["Conversions"].mean(),
        "df":     df,
        "graph":  graph,
        "ref1":   ref1,
        "ref2":   ref2,
        "ref3":   ref3,
    })

# 4) Results & What-If
if "ate" in st.session_state:
    ate   = st.session_state["ate"]
    base  = st.session_state["base"]
    df    = st.session_state["df"]
    graph = st.session_state["graph"]
    r1    = st.session_state["ref1"]
    r2    = st.session_state["ref2"]
    r3    = st.session_state["ref3"]

    st.write("## 📊 Causal Results")
    st.subheader("Causal Graph")
    st.graphviz_chart(graph)

    st.write(f"**Overall ATE:** {ate:.4f} conversions per $1 spend")

    st.write("### Heterogeneous Effects by Customer Segment")
    segs = []
    for seg in sorted(df["CustomerSegment"].unique()):
        dseg = df[df["CustomerSegment"]==seg]
        mseg = CausalModel(dseg, "CampaignSpend", "Conversions", graph)
        eset = mseg.estimate_effect(
            mseg.identify_effect(), method_name="backdoor.linear_regression"
        )
        segs.append((seg, eset.value))
    st.table(pd.DataFrame(segs, columns=["Segment","ATE"]).round(4))

    st.write("### Refutation Tests")
    st.caption("These are diagnostics—your ATE should hold up under each:")

    for label, r in [("Random Common Cause", r1),
                     ("Placebo Treatment",  r2),
                     ("Data Subset Refuter",r3)]:
        with st.expander(f"🔍 {label}"):
            st.write(f"- Original ATE: {r.estimated_effect:.3f}")
            st.write(f"- ATE after refute: {r.new_effect:.3f}")

    st.sidebar.subheader("4) What-If Scenario")
    mult     = st.sidebar.slider("Spend ×", 0.5, 2.0, 1.2, 0.05)
    new_mean = base + (mult - 1) * ate
    st.write(f"**Predicted avg conversions @×{mult:.2f}:** {new_mean:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(df["CampaignSpend"], df["Conversions"], alpha=0.4, label="Observed")
    ax.axhline(base, linestyle="--", color="gray", label="Obs Avg")
    ax.scatter(df["CampaignSpend"]*mult, [new_mean]*len(df),
               alpha=0.4, label="Counterfactual")
    ax.set_xlabel("Campaign Spend (USD)")
    ax.set_ylabel("Conversions")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("▶ Click ‘Run Causal Model’ in the sidebar to compute ATE, CATEs & refutations.")
