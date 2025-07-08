# 📘 App Tutorial: Ontology + Causal AI Demo

**Welcome!** This guide walks you through every panel of the dashboard—what you see, why it matters, and how to interpret each live number.

---

## 🔄 Controls (Left Sidebar)

- **📖 Show Tutorial**  
  Toggle on to display this guide. Uncheck to return to the main app.

- **🔄 Reset App**  
  Clears all computed results so you can start fresh.

---

## 1) Data

- **Synthetic** (default):  
  Instantly generates a “fake” marketing funnel:
  - **CampaignSpend** (USD): uniform \$100–\$1000  
  - **Seasonality**: binary 0/1 (adds \$200 spend when 1)  
  - **AdQuality**: “Low”/“High” (High gives +15% spend & +3 conv)  
  - **ChannelType**: Search/Social (–10% spend) / Display (–2 conv)  
  - **CustomerSegment**: A (+0.05 conv/\$) or B (+0.02 conv/\$)  
  - **Impressions → Clicks → Conversions**: with noise & confounder effects  

- **Upload**: drop a CSV/ZIP with columns  
  `CampaignSpend, Seasonality, AdQuality, ChannelType, CustomerSegment, Impressions, Clicks, Conversions`

Use **Samples** slider and **Seed** values for size/reproducibility.

---

## 2) Ontology

- **Built-In**: shows our sample `marketing_ontology.owl` classes:  
  `Campaign`, `Impression`, `Click`, `Conversion`  
- **Upload**: drop your own OWL file and see its classes listed.

> Ontology class names (e.g. “Campaign”) are your domain concepts; the DAG nodes use the actual data field names (e.g. `CampaignSpend`).

---

## 3) Run Causal Model

Click ▶️ **Run Causal Model** to:

1. **Render the DAG**  
   - Confounders (Seasonality, AdQuality, ChannelType, CustomerSegment) point into both  
     `CampaignSpend` and `Conversions`.  
   - The funnel chain is `CampaignSpend → Impressions → Clicks → Conversions`.

2. **Compute** the **Average Treatment Effect (ATE)** of spend on conversions  
   (in **conversions per \$1**), via back-door linear regression.

All results live in **session state** so you can still move the **Spend ×** slider afterward without re-running.

---

# 📊 Causal Results

### Causal Graph  
An inline Graphviz chart of your DAG, confirming which arrows (edges) were adjusted for.

### Overall ATE  
**Conversions per \$1 spend**  
- Example: **0.0843** means *0.0843 extra conversions* per extra dollar.

---

## Heterogeneous Effects by Customer Segment

| Segment | ATE (conv/\$) |
|:-------:|:-------------:|
| A       | 0.1037        |
| B       | 0.0744        |

*These numbers update live as you move the “Spend ×” slider, showing any slight shifts in segment-level effects under different what-if budgets.*

---

## 4) What-If Scenario

- **Spend ×** slider (0.5–2.0): simulate halving/doubling your budget  
- **Predicted avg conversions** =  
  `observed_mean + (multiplier − 1) × ATE`  
- **Scatter plot** overlays:  
  - **Blue dots** = observed `(Spend, Conversions)`  
  - **Gray dashed line** = observed mean conversions  
  - **Orange dots** = counterfactual conversions at your chosen multiplier  

> **Axes:** X = Campaign Spend (USD), Y = Conversions (count)

---

## 🧭 Why It Matters

- **Causal vs. Correlation**: you explicitly adjust for confounders.  
- **Actionable “What-if”**: predict outcomes under real interventions.  
- **Dynamic insights**: see how small budget changes ripple through overall and segment-level estimates.

Enjoy exploring causal AI with confidence! 🎉
