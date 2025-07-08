import numpy as np
import pandas as pd
import os
import zipfile

def generate_data(n=500, seed=0,
                  output_csv="data/marketing_data.csv",
                  zip_path="data/marketing_data.zip"):
    np.random.seed(seed)

    # 1) Base spend
    df = pd.DataFrame({
        "CampaignSpend": np.random.uniform(100, 1000, n)
    })

    # 2) Seasonality confounder
    df["Seasonality"] = np.random.choice([0, 1], size=n)
    df["CampaignSpend"] += df["Seasonality"] * 200

    # 3) AdQuality & ChannelType confounders
    df["AdQuality"] = np.random.choice(["Low","High"], size=n)
    # High-quality ads get +15% spend
    df["CampaignSpend"] += (df["AdQuality"] == "High") * 0.15 * df["CampaignSpend"]

    df["ChannelType"] = np.random.choice(["Search","Social","Display"], size=n)
    # Social channels are −10% spend efficiency
    df["CampaignSpend"] *= np.where(df["ChannelType"]=="Social", 0.90, 1.0)

    # 4) Customer segment for heterogeneous effects
    df["CustomerSegment"] = np.random.choice(["A", "B"], size=n)
    segment_effect = np.where(df["CustomerSegment"] == "A", 0.05, 0.02)

    # 5) Funnel
    df["Impressions"] = df["CampaignSpend"] * 10 + np.random.normal(0, 100, n)
    df["Clicks"]      = df["Impressions"] * 0.05 + np.random.normal(0, 5, n)
    df["Conversions"] = (
        df["Clicks"] * 0.1
        + np.random.normal(0, 2, n)
        + df["Seasonality"] * 5
        + df["CampaignSpend"] * segment_effect
        # AdQuality & ChannelType effects on conversions
        + (df["AdQuality"] == "High") * 3
        - (df["ChannelType"] == "Display") * 2
    )

    # Save & zip
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(output_csv, arcname=os.path.basename(output_csv))

    print(f"Synthetic data saved to {output_csv} and zipped to {zip_path}")

if __name__ == "__main__":
    generate_data()
