import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("mem-bench.out", delim_whitespace=True)
data_tens = data[data["Nrow"] % 10 == 0]
data_non_tens = data[data["Nrow"] % 10 != 0]

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Nrow (log scale)')
ax.set_ylabel('Effective BW (log scale)')
ax.set_title('Memory Benchmark Results')
colors = ['r', 'g']
for i, data in enumerate([data_tens, data_non_tens]):
    ax.plot(data["Nrow"], data["EffectiveBW"], marker='o', linestyle='-', color=colors[i], label='Multiple of 10' if i == 0 else 'Multiple of 2^x')

ax.legend()
plt.show()
fig.savefig("mem-bench-plot.png", dpi=300)