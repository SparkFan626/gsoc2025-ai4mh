import pandas as pd
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import time
import os

# Create output directory if not exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Initialize pytrends request
pytrends = TrendReq(hl='en-US', tz=360)

# Define keywords to track on Google Trends
keywords = ["depression", "suicide", "loneliness", "mental health", "anxiety"]

# Define time range for the trend data (1 year period)
timeframe = '2024-06-01 2025-06-01'

# Fetch trend data keyword by keyword to avoid API rate limiting
all_trends = pd.DataFrame()

for kw in keywords:
    pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo='', gprop='')
    df = pytrends.interest_over_time()
    if not df.empty:
        df = df[[kw]]
        df.columns = [kw]
        all_trends = pd.concat([all_trends, df], axis=1)
    time.sleep(2)  # Sleep between requests to avoid being blocked

# Save trend data to CSV
csv_path = os.path.join(output_dir, "google_trends_mental_health.csv")
all_trends.to_csv(csv_path)

# Plot and save figure
plt.figure(figsize=(12, 6))
all_trends.plot(title="Google Trends (2024.06 - 2025.06)", figsize=(12, 6))
plt.ylabel("Trend Score")
plt.xlabel("Date")
plt.tight_layout()
plot_path = os.path.join(output_dir, "google_trends_plot.png")
plt.savefig(plot_path)
plt.close()

csv_path, plot_path
