# --- Full stub of dowhy.causal_refuters & numpy.distutils to avoid EconML/distutils errors ---
import sys, types

# 1) Stub the root refuters package so imports like `import dowhy.causal_refuters` work
_ref_pkg = types.ModuleType("dowhy.causal_refuters")
sys.modules["dowhy.causal_refuters"] = _ref_pkg

# 2) Stub each submodule DoWhy may import, and define the minimal symbols they expect
def _make_stub(name, attrs):
    m = types.ModuleType(name)
    for attr, val in attrs.items():
        setattr(m, attr, val)
    sys.modules[name] = m

# a) graph_refuter needs GraphRefuter
_make_stub(
    "dowhy.causal_refuters.graph_refuter",
    {"GraphRefuter": type("GraphRefuter", (), {"__init__": lambda *a, **k: None})}
)

# b) add_unobserved_common_cause expects AddUnobservedCommonCause, sensitivity_e_value, sensitivity_simulation
_make_stub(
    "dowhy.causal_refuters.add_unobserved_common_cause",
    {
        "AddUnobservedCommonCause": type("AddUnobservedCommonCause", (), {"__init__": lambda *a, **k: None}),
        "sensitivity_e_value":     lambda *a, **k: None,
        "sensitivity_simulation":  lambda *a, **k: None,
    },
)

# c) random_common_cause refuter
_make_stub(
    "dowhy.causal_refuters.random_common_cause",
    {"RandomCommonCause": type("RandomCommonCause", (), {"__init__": lambda *a, **k: None})}
)

# d) placebo_treatment_refuter
_make_stub(
    "dowhy.causal_refuters.placebo_treatment_refuter",
    {"PlaceboTreatmentRefuter": type("PlaceboTreatmentRefuter", (), {"__init__": lambda *a, **k: None})}
)

# e) data_subset_refuter
_make_stub(
    "dowhy.causal_refuters.data_subset_refuter",
    {"DataSubsetRefuter": type("DataSubsetRefuter", (), {"__init__": lambda *a, **k: None})}
)

# 3) Stub out numpy.distutils.misc_util so EconML import won’t crash
_misc = types.ModuleType("numpy.distutils.misc_util")
_misc.is_sequence = lambda x: isinstance(x, (list, tuple))
sys.modules["numpy.distutils.misc_util"] = _misc
# --- End stubs ---

# --- monkey‐patch for DoWhy + NetworkX compatibility ---
import networkx as _nx
from networkx.algorithms.d_separation import d_separated as _dsep
_nx.algorithms.d_separated = _dsep
# --- end patch ---

import os, io, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from owlready2 import get_ontology
from dowhy import CausalModel   # Now safe: refuters will be no-ops

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
if ont_mode == "Built-In":
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

    # three refutation tests
    ref1 = model.refute_estimate(estimand, estimate,
                                 method_name="random_common_cause")
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
        dseg   = df[df["CustomerSegment"]==seg]
        mseg   = CausalModel(dseg, "CampaignSpend", "Conversions", graph)
        eset   = mseg.estimate_effect(
            mseg.identify_effect(), method_name="backdoor.linear_regression"
        )
        segs.append((seg, eset.value))
    st.table(pd.DataFrame(segs,columns=["Segment","ATE"]).round(4))

    st.write("### Refutation Tests")
    st.caption("These are diagnostics—your ATE should hold up under each:")

    with st.expander("🔍 Random Common Cause"):
        st.write(f"- Original ATE: {r1.estimated_effect:.3f}")
        st.write(f"- ATE after refute: {r1.new_effect:.3f}")

    with st.expander("🔍 Placebo Treatment"):
        st.write(f"- Original ATE: {r2.estimated_effect:.3f}")
        st.write(f"- ATE after refute: {r2.new_effect:.3f}")

    with st.expander("🔍 Data Subset Refuter"):
        st.write(f"- Original ATE: {r3.estimated_effect:.3f}")
        st.write(f"- ATE after refute: {r3.new_effect:.3f}")

    st.sidebar.subheader("4) What-If Scenario")
    mult     = st.sidebar.slider("Spend ×", 0.5, 2.0, 1.2, 0.05)
    new_mean = base + (mult - 1) * ate
    st.write(f"**Predicted avg conversions @×{mult:.2f}:** {new_mean:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(df["CampaignSpend"], df["Conversions"],
               alpha=0.4, label="Observed")
    ax.axhline(base, linestyle="--", color="gray", label="Obs Avg")
    ax.scatter(df["CampaignSpend"]*mult,
               [new_mean]*len(df),
               alpha=0.4, label="Counterfactual")
    ax.set_xlabel("Campaign Spend (USD)")
    ax.set_ylabel("Conversions")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("▶ Click ‘Run Causal Model’ in the sidebar to compute ATE, CATEs & refutations.")
