#!/usr/bin/env python3
"""metrics_visualization.py

Standalone script to render a grouped‑bar chart of four evaluation metrics
(Balanced Accuracy, Cohen’s Kappa, Weighted F1, macro ROC‑AUC) for three models:
TabNet, Logistic Regression, and XGBoost.

Run:
    python metrics_visualization.py
The chart will pop up in an interactive window if you have a display, or be
saved to PNG when running head‑less (e.g. on a cluster).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------------
# Hard‑coded results from the latest evaluation run
# -----------------------------------------------------------------------------
models = ["TabNet", "LogReg", "XGBoost"]
metrics = ["BalAcc", "Kappa", "F1_w", "AUC"]
# Scores are from the latest evaluation run sourced from results/metrics/all_metrics_TIME_STAMP.txt
# Later, we will read the scores from the file
scores = {
    "TabNet":  [0.3320, -0.0029, 0.1298, 0.4986],
    "LogReg":  [0.7033, 0.5393, 0.7374, 0.8610],
    "XGBoost": [0.9059, 0.8494, 0.9125, 0.9778],
}

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
bar_w = 0.25
x = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(8, 4))

for i, model in enumerate(models):
    ax.bar(x + i * bar_w, scores[model], width=bar_w, label=model)

ax.set_xticks(x + bar_w)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison Across Evaluation Metrics")
ax.legend()
plt.tight_layout()

# Detect head‑less mode
if os.environ.get("DISPLAY", "") == "":
    out_png = os.path.expanduser("~/uiuc-cancer-research/results/metrics/metrics_grouped_bar.png")
    plt.savefig(out_png, dpi=300)
    print(f"📊 Figure saved to {out_png}")
else:
    plt.show()
