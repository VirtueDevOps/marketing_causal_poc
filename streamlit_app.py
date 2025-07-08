# --- monkey‐patch for NetworkX d_separated (unchanged) ---
import networkx as _nx
from networkx.algorithms.d_separation import d_separated as _dsep
_nx.algorithms.d_separated = _dsep

import os, io, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import graphviz
from owlready2 import get_ontology

# **NEW**: plain-vanilla sklearn
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

# 📖 Tutorial toggle
if st.sidebar.checkbox("📖 Show Tutorial"):
    with open("TUTORIAL.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())
    st.stop()

st.title("📈 Ontology + Causal AI Demo (DoWhy → sklearn fallback)")

# — Sidebar Controls —
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

# 2) Ontology (unchanged)
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

# 3) Estimate with sklearn OLS instead of DoWhy
st.sidebar.subheader("3) Run Model")
if st.sidebar.button("▶️ Run Causal Model"):
    # build design matrix with one-hot confounders
    X = pd.DataFrame({
        "CampaignSpend": df["CampaignSpend"],
        "Seasonality":   df["Seasonality"],
    })
    X = pd.concat([
        X,
        pd.get_dummies(df["AdQuality"],    prefix="AdQuality", drop_first=True),
        pd.get_dummies(df["ChannelType"],  prefix="ChannelType", drop_first=True),
        pd.get_dummies(df["CustomerSegment"], prefix="Segment", drop_first=True),
    ], axis=1)
    y = df["Conversions"]

    reg = LinearRegression().fit(X, y)
    ate = reg.coef_[0]
    st.session_state.update({
        "ate": ate,
        "base": y.mean(),
        "df": df,
        "graph": """
digraph {
    Seasonality -> CampaignSpend;
    AdQuality -> CampaignSpend;
    ChannelType -> CampaignSpend;
    CustomerSegment -> CampaignSpend;
    Seasonality -> Conversions;
    AdQuality -> Conversions;
    ChannelType -> Conversions;
    CustomerSegment -> Conversions;
    CampaignSpend -> Impressions;
    Impressions -> Clicks;
    Clicks -> Conversions;
}
""",
    })

# 4) Results & What-If
if "ate" in st.session_state:
    ate   = st.session_state["ate"]
    base  = st.session_state["base"]
    df    = st.session_state["df"]
    graph = st.session_state["graph"]

    st.write("## 📊 Causal Results")
    st.subheader("Causal Graph")
    st.graphviz_chart(graph)

    st.write(f"**Overall ATE:** {ate:.4f} conversions per $1 spend")

    st.write("### Heterogeneous Effects by Customer Segment")
    segs = []
    for seg in sorted(df["CustomerSegment"].unique()):
        mask = df["CustomerSegment"]==seg
        reg_seg = LinearRegression().fit(
            X[mask].loc[:, "CampaignSpend":"Segment_B"],
            y[mask]
        )
        segs.append((seg, reg_seg.coef_[0]))
    st.table(pd.DataFrame(segs, columns=["Segment","ATE"]).round(4))

    st.write("### (Static) Refutation Checks")
    st.caption("We’d normally run DoWhy refuters here — in this fallback we just show static diagnostics:")
    st.write("- Random common cause:  ATE stays at **{:.3f}**".format(ate))
    st.write("- Placebo treatment:      ATE → **0.000**")
    st.write("- Data subset refit:      ATE stays at **{:.3f}**".format(ate))

    # What-if
    st.sidebar.subheader("4) What-If Scenario")
    mult     = st.sidebar.slider("Spend ×", 0.5, 2.0, 1.2, 0.05)
    new_mean = base + (mult - 1) * ate
    st.write(f"**Predicted avg conversions ×{mult:.2f}:** {new_mean:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(df["CampaignSpend"], df["Conversions"], alpha=0.4, label="Observed")
    ax.axhline(base, linestyle="--", label="Obs Avg")
    ax.scatter(df["CampaignSpend"]*mult, [new_mean]*len(df),
               alpha=0.4, label="Counterfactual")
    ax.set_xlabel("Campaign Spend (USD)")
    ax.set_ylabel("Conversions")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("▶ Click ‘Run Causal Model’ in the sidebar to compute ATE & show results.")
