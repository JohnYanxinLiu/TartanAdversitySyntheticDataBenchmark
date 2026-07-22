import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
RESULTS_DIR = "final_results"
OUTPUT_PATH = "performance_trends_extra_large.pdf"

MODEL_TYPES = ['resnet', 'convnext', 'yolo']
BDD_METRIC = 'val/clear/mAP'
DAWN_METRICS = {
    'dawn/rainy/mAP': 'Rain',
    'dawn/snowy/mAP': 'Snow',
    'dawn/foggy/mAP': 'Fog',
}

# Metric Styling
colors = {BDD_METRIC: '#2E7D32', 'dawn/rainy/mAP': '#1976D2', 'dawn/snowy/mAP': '#7B1FA2', 'dawn/foggy/mAP': '#D32F2F'}
markers = {BDD_METRIC: 'o', 'dawn/rainy/mAP': 's', 'dawn/snowy/mAP': '^', 'dawn/foggy/mAP': 'D'}

# ==========================================
# 2. IEEE/IROS PLOT SETTINGS - EXTRA LARGE
# ==========================================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,               # Base font size
    "axes.labelsize": 18,          # X and Y labels
    "axes.titlesize": 18,          # Subplot titles
    "xtick.labelsize": 14,         # Tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 16,         # Legend text
    "text.usetex": False
})

# ==========================================
# 3. PLOTTING LOGIC
# ==========================================
# Increased figsize width to 18 to handle the extra x-ticks comfortably
fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex='col', sharey='row')

for r_idx, model in enumerate(MODEL_TYPES):
    for c_idx, (mix_type, aug_type) in enumerate([
        ('r', 'automold'), ('r', 'gemini'), 
        ('a', 'automold'), ('a', 'gemini')
    ]):
        ax = axes[r_idx, c_idx]
        
        filepath = os.path.join(RESULTS_DIR, f"{model}_{mix_type}_results.csv")
        if not os.path.exists(filepath):
            ax.text(0.5, 0.5, "N/A", ha='center', fontsize=16)
            continue
            
        df = pd.read_csv(filepath)
        data = df[df['aug_type'] == aug_type].sort_values('mix_value')

        if data.empty:
            continue

        # Plot BDD Clear
        ax.plot(data['mix_value'], data[BDD_METRIC], 
                label='BDD Clear', color=colors[BDD_METRIC], 
                marker=markers[BDD_METRIC], linewidth=2.5, markersize=8)

        # Plot DAWN Metrics
        for metric, label in DAWN_METRICS.items():
            if metric in data.columns:
                ax.plot(data['mix_value'], data[metric], 
                        label=label, color=colors[metric], 
                        marker=markers[metric], linestyle='--', 
                        linewidth=2.0, markersize=8, alpha=0.8)

        # Titles and Labels
        if r_idx == 0:
            title_prefix = "REPLACEMENT" if mix_type == 'r' else "ADDITIVE"
            ax.set_title(f"{title_prefix}\n{aug_type.upper()}", pad=15)
        
        if c_idx == 0:
            ax.set_ylabel(f"{model.upper()}\nmAP (%)", labelpad=10)

        if r_idx == 2:
            ax.set_xlabel("Mix %", labelpad=10)

        # UPDATED TICK LOGIC
        if mix_type == 'r':
            # Requested: 10, 20, 30, 40, 50 (adding 0 for baseline completeness if exists)
            ax.set_xticks([0, 10, 20, 30, 40, 50])
        else:
            # Requested: 10, 20, 30
            ax.set_xticks([0, 10, 20, 30])
            
        ax.grid(True, linestyle=':', alpha=0.6)

# Handle Global Legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, 
            bbox_to_anchor=(0.5, -0.05), frameon=True, borderpad=1.2)

# Spacing adjustment for extra large text and labels
plt.subplots_adjust(wspace=0.15, hspace=0.45)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300)
print(f"Extra-large font figure with updated ticks saved to {OUTPUT_PATH}")
plt.show()