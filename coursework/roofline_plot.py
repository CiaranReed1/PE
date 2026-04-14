import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User-selectable options
# -----------------------------
use_median = False   # True = median, False = mean
AI =  0.2443            # Arithmetic intensity (FLOP/Byte)

# -----------------------------
# Hardware parameters
# -----------------------------
memory_bandwidth = 24243.48e6        # B/s
scalar_compute_roof = 1.34e10   # FLOP/s (scalar)
avx_compute_roof = 4 * scalar_compute_roof  # FLOP/s (AVX, 4-wide)

# -----------------------------
# Plot setup
# -----------------------------
AI_range = np.logspace(-2, 2, 400)
fig, ax = plt.subplots(figsize=(10, 6))


# -----------------------------
# Plot measured performance
# -----------------------------
flops = 606.4646e6
ax.scatter(AI,flops, color='red', label='Measured Performance', zorder=5,marker ="x")
# -----------------------------
# Roofline models (min-based)
# -----------------------------

# Scalar roofline
ax.loglog(
    AI_range,
    np.minimum(memory_bandwidth * AI_range, scalar_compute_roof),
    linewidth=2,
    label='Scalar Roofline '+str(scalar_compute_roof/1e9)+' GFLOP/s'
)

# AVX roofline
ax.loglog(
    AI_range,
    np.minimum(memory_bandwidth * AI_range, avx_compute_roof),
    linewidth=2,
    label='AVX Roofline (4-wide) '+str(avx_compute_roof/1e9)+' GFLOP/s'
)

# -----------------------------
# Plot formatting
# -----------------------------
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-1, 1e1)
ax.set_ylim(1e8, 1e12)

ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax.set_ylabel('Performance (FLOP/s)')
ax.legend(loc='best')


plt.tight_layout()
plt.savefig('coursework/roofline_plot.png', dpi=300)
plt.show()
 