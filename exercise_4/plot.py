import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User-selectable options
# -----------------------------
use_median = False   # True = median, False = mean
AI = 0.25            # Arithmetic intensity (FLOP/Byte)

# -----------------------------
# Hardware parameters
# -----------------------------
memory_bandwidth = 19017.15e6        # B/s
scalar_compute_roof = 53.6e9         # FLOP/s (scalar)
avx_compute_roof = 4 * scalar_compute_roof  # FLOP/s (AVX, 4-wide)

# -----------------------------
# Plot setup
# -----------------------------
AI_range = np.logspace(-2, 2, 400)
fig, ax = plt.subplots(figsize=(10, 6))

blocked = ["", "-blocked"]
optimisation_levels = ["O0", "O1", "O2", "O3"]
colors = [
    "red", "blue", "green", "orange",
    "purple", "brown", "pink", "gray"
]

# -----------------------------
# Plot measured performance
# -----------------------------
i = 0
for block in blocked:
    for level in optimisation_levels:
        data = pd.read_csv(
            f'roofline{block}_{level}.dat',
            delim_whitespace=True
        )

        perf = (
            data["flops"].median()
            if use_median
            else data["flops"].mean()
        )

        ax.scatter(
            AI,
            perf * 1e6,  # MFLOP/s → FLOP/s
            color=colors[i],
            marker="o" if block == "" else "x",
            s=80,
            label=f'{level}{block}'
        )

        i += 1

# -----------------------------
# Roofline models (min-based)
# -----------------------------

# Scalar roofline
ax.loglog(
    AI_range,
    np.minimum(memory_bandwidth * AI_range, scalar_compute_roof),
    linewidth=2,
    label='Scalar Roofline'
)

# AVX roofline
ax.loglog(
    AI_range,
    np.minimum(memory_bandwidth * AI_range, avx_compute_roof),
    linewidth=2,
    label='AVX Roofline (4-wide)'
)

# -----------------------------
# Plot formatting
# -----------------------------
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-1, 1e2)
ax.set_ylim(1e8, 1e12)

ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax.set_ylabel('Performance (FLOP/s)')
ax.set_title(
    f'Roofline Plot ({ "Median" if use_median else "Mean" } Performance)'
)

ax.legend(loc='best')


plt.tight_layout()
plt.savefig('roofline_plot.png', dpi=300)
plt.show()
