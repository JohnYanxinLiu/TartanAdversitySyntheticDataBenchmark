import matplotlib.pyplot as plt

thresholds = [0, 10, 15, 20]

data = {
    "Snow": {
        "Automold": {
            "FID": [43.7, 38.9, 37.3, 36.0],
            "CMMD": [2.61, 2.27, 2.14, 2.03]
        },
        "Gemini": {
            "FID": [54.6, 50.3, 48.7, 47.2],
            "CMMD": [3.90, 3.58, 3.46, 3.32]
        }
    },
    "Fog": {
        "Automold": {
            "FID": [241.9, 236.0, 232.9, 230.0],
            "CMMD": [14.18, 13.91, 13.71, 13.51]
        },
        "Gemini": {
            "FID": [174.5, 168.9, 166.2, 163.7],
            "CMMD": [8.13, 7.85, 7.68, 7.52]
        }
    },
    "Rain": {
        "Automold": {
            "FID": [91.5, 84.6, 81.8, 79.0],
            "CMMD": [6.80, 6.35, 6.13, 5.91]
        },
        "Gemini": {
            "FID": [67.4, 62.8, 61.2, 59.6],
            "CMMD": [4.56, 4.44, 4.34, 4.22]
        }
    }
}

fig, axes = plt.subplots(2, 3, figsize=(14,7))

weather_list = ["Snow", "Fog", "Rain"]

for col, weather in enumerate(weather_list):

    # -------- FID --------
    ax = axes[0, col]

    for method in data[weather]:
        ax.plot(thresholds, data[weather][method]["FID"], marker='o', label=method)

    ax.set_title(weather)
    ax.set_ylabel("FID ↓")
    ax.set_xticks(thresholds)
    ax.grid(True)

    fid_vals = []
    for m in data[weather]:
        fid_vals += data[weather][m]["FID"]

    fid_min, fid_max = min(fid_vals), max(fid_vals)
    margin = (fid_max - fid_min) * 0.2
    ax.set_ylim(fid_min - margin, fid_max + margin)

    ax.legend()

    # -------- CMMD --------
    ax = axes[1, col]

    for method in data[weather]:
        ax.plot(thresholds, data[weather][method]["CMMD"], marker='s')

    ax.set_ylabel("CMMD ×10² ↓")
    ax.set_xlabel("Pruning Threshold (%)")
    ax.set_xticks(thresholds)
    ax.grid(True)

    cmmd_vals = []
    for m in data[weather]:
        cmmd_vals += data[weather][m]["CMMD"]

    cmmd_min, cmmd_max = min(cmmd_vals), max(cmmd_vals)
    margin = (cmmd_max - cmmd_min) * 0.2
    ax.set_ylim(cmmd_min - margin, cmmd_max + margin)

plt.suptitle("Distributional Fidelity Across Pruning Thresholds", fontsize=18)

plt.tight_layout()

plt.savefig("distributional_fidelity_grid.png", dpi=300, bbox_inches="tight")
plt.savefig("distributional_fidelity_grid.pdf", bbox_inches="tight")

plt.show()