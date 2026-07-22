import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PATHS CONFIGURATION
# ==========================================
# Synthetic-data ranking CSVs, relative to the repo's Data/ directory
# (override the root with the DATA_DIR env var if your data lives elsewhere).
DATA_DIR = os.getenv("DATA_DIR", "Data")
csv_paths = {
    "Automold-Rain": os.path.join(DATA_DIR, "AutomoldAugmented/AutomoldRankings/rain.csv"),
    "Automold-Fog":  os.path.join(DATA_DIR, "AutomoldAugmented/AutomoldRankings/fog.csv"),
    "Automold-Snow": os.path.join(DATA_DIR, "AutomoldAugmented/AutomoldRankings/snow.csv"),
    "Gemini-Rain":   os.path.join(DATA_DIR, "GeminiAugmented/GeminiRankings/rain.csv"),
    "Gemini-Fog":    os.path.join(DATA_DIR, "GeminiAugmented/GeminiRankings/fog.csv"),
    "Gemini-Snow":   os.path.join(DATA_DIR, "GeminiAugmented/GeminiRankings/snow.csv"),
}

# ==========================================
# 2. EXTRA LARGE PLOT SETTINGS
# ==========================================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,               # Base font size
    "axes.labelsize": 20,          # X and Y label size
    "axes.titlesize": 20,          # Subplot titles
    "xtick.labelsize": 16,         # Axis numbers
    "ytick.labelsize": 16,
    "legend.fontsize": 18,         # Legend font
    "text.usetex": False 
})

plot_order = ["Automold-Rain", "Automold-Fog", "Automold-Snow", 
              "Gemini-Rain", "Gemini-Fog", "Gemini-Snow"]
colors = {"Automold": "#4C72B0", "Gemini": "#55A868"}

# ==========================================
# 3. GENERATE GRID
# ==========================================
# Increased height (11) to give the labels breathing room
fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
axes = axes.flatten()

handles, labels = [], []

for i, label in enumerate(plot_order):
    ax = axes[i]
    method = "Automold" if "Automold" in label else "Gemini"
    
    try:
        df = pd.read_csv(csv_paths[label])
        scores = df['metric_score']
        
        sns.histplot(scores, kde=True, ax=ax, 
                     color=colors[method], edgecolor='white', alpha=0.5, stat="density")
        
        threshold = np.percentile(scores, 80)
        
        # Thicker line for better visibility at high DPI
        ax.axvline(threshold, color='darkred', linestyle='--', linewidth=3.5,
                   label=r'$\alpha=20\%$ cutoff')
        
        ax.axvspan(threshold, scores.max(), color='gray', alpha=0.15, 
                   label='Filtered Region (Top 20%)')

        ax.set_title(label, fontweight='bold', pad=20)
        
        if i == 0:
            curr_handles, curr_labels = ax.get_legend_handles_labels()
            handles.extend(curr_handles)
            labels.extend(curr_labels)
        
    except FileNotFoundError:
        ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', fontsize=20)

    # Added more padding (labelpad) so text doesn't touch the axes
    if i >= 3: 
        ax.set_xlabel(r"Fidelity Score $h^{\star}(x, \tilde{x})$", labelpad=20)
    if i % 3 == 0: 
        ax.set_ylabel("Density", labelpad=20)

# --- GLOBAL LEGEND AT BOTTOM ---
# bbox_to_anchor moved slightly lower to clear the X-axis labels
fig.legend(handles, labels, loc='lower center', ncol=2, 
           bbox_to_anchor=(0.5, -0.02), frameon=True, borderpad=1.2)

# Subplots_adjust ensures the titles of the bottom row don't hit the top row
plt.subplots_adjust(wspace=0.15, hspace=0.4)
plt.tight_layout(rect=[0, 0.05, 1, 0.98]) 

plt.savefig('fidelity_distributions_MAX_font.pdf', bbox_inches='tight', dpi=300)
plt.show()