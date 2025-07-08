# 📘 App Tutorial: Ontology + Causal AI Demo

**Welcome!** This guide walks you through every panel of the dashboard—what you see, why it matters, and how to interpret each result, including the example numbers you’ll encounter (e.g. 0.0843, 0.0000).

---

## 🔄 Controls (Left Sidebar)

- **📖 Show Tutorial**  
  Toggle on to display this guide. Uncheck to return to the main app.

- **🔄 Reset App**  
  Clears all computed results (ATE, CATEs, diagnostics) so you can start fresh.

---

## 1) Data

### Modes

- **Synthetic** (default)  
  Generates a “fake” marketing funnel with:
  1. **CampaignSpend** (USD): Uniform \$100–\$1000  
  2. **Seasonality**: 0/1 confounder (adds \$200 spend when 1)  
  3. **AdQuality**: “Low”/“High” (High adds +15% spend & +3 conversions)  
  4. **ChannelType**: Search/Social (–10% spend) /Display (–2 conversions)  
  5. **CustomerSegment**: A (+0.05 conv/\$) or B (+0.02 conv/\$)  
  6. **Impressions → Clicks → Conversions**: with noise + confounder & segment effects

- **Upload**  
  Supply a ZIP/CSV with columns:  
  `CampaignSpend, Seasonality, AdQuality, ChannelType, CustomerSegment, Impressions, Clicks, Conversions`

### Controls

- **Samples**: Number of rows to generate (100–5000)  
- **Seed**: Random seed for reproducibility

---

## 2) Ontology

- **Built-In**: Loads `marketing_ontology.owl` (classes: Campaign, Impression, Click, Conversion)  
- **Upload**: Drop your own OWL; its classes will be listed

> **Domain vs. Data names:**  
> Ontology classes are abstract concepts (e.g. “Campaign”), whereas DAG nodes are measured variables (`CampaignSpend`, `Impressions`, etc.).

---

## 3) Run Causal Model

Click **▶️ Run Causal Model** to:

1. **Draw the DAG**  
   - Confounders → `CampaignSpend` & `Conversions`  
     (`Seasonality`, `AdQuality`, `ChannelType`, `CustomerSegment`)  
   - Funnel chain: `CampaignSpend → Impressions → Clicks → Conversions`

2. **Identify** the **Average Treatment Effect (ATE)** of spend on conversions, adjusting for confounders.

3. **Estimate** via back-door linear regression.

4. **Diagnostics** (DoWhy refutation tests):  
   - **Random Common Cause**: add noise; ATE should stay the same  
   - **Placebo Treatment**: permute spend; ATE should collapse to ~0.0000  
   - **Data Subset Refuter**: fit on 70% of data; ATE should remain ~0.0843

All results live in **session state** so you can tweak sliders without re-running.

---

# 📊 Causal Results

### Causal Graph  
An inline Graphviz chart showing the DAG you adjusted for.

### Overall ATE  
**Conversions per \$1 spend**  
- Example: **0.0843** means *0.0843 extra conversions* per additional dollar spent.

---

## Heterogeneous Effects by Customer Segment

| Segment | ATE (conv/\$) |
|:-------:|:-------------:|
| A       | 0.0995        |
| B       | 0.0705        |

Segment A responds more strongly (0.0995) than B (0.0705).

---

## Refutation Tests

These **expanders** let you inspect each diagnostic:

<details>
<summary>🔍 Random Common Cause</summary>

- **Original ATE:** 0.0843  
- **After adding random cause:** 0.0843  

*Shows adding noise doesn’t alter the effect.*
</details>

<details>
<summary>🔍 Placebo Treatment</summary>

- **Original ATE:** 0.0843  
- **After permuting spend:** 0.0000  

*Confirms the effect vanishes when no true treatment exists.*
</details>

<details>
<summary>🔍 Data Subset Refuter</summary>

- **Original ATE:** 0.0843  
- **On 70% subset:** 0.0844  

*Demonstrates stability across random subsamples.*
</details>

> **Why these matter:**  
> - Near-identical ATEs (e.g. 0.0843→0.0843) confirm robustness.  
> - Placebo ~0.0000 confirms no spurious signals.  
> - Small random shifts (±0.0001) are expected; large swings would be a red flag.

---

## 4) What-If Scenario

- **Spend ×** slider (0.5–2.0): scale budget down/up  
- **Predicted avg conversions** = `observed_mean + (mult − 1) × ATE`  
- **Scatter Plot**  
  - **Blue**: observed data  
  - **Gray dashed line**: observed mean  
  - **Orange**: counterfactual at chosen multiplier  

> **Axes:** X = CampaignSpend (USD), Y = Conversions (count)

---

## 🧭 Why It Matters

- **Causal vs. Correlation:** adjusts for confounders instead of just correlating.  
- **Actionable “What-if” insights:** predict outcomes under interventions.  
- **Trust but verify:** built-in diagnostics stress-test every claim.

Enjoy exploring causal AI for robust, interpretable decision-making! 🎉
