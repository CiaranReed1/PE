import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Nrow (log scale)')
ax.set_ylabel('Effective BW (log scale) (MBytes/s)')
ax.set_title('Memory Benchmark Results')
colors = ['r', 'g','b']
linestyles = "-","--"
for j, source in enumerate(["transpose","transpose-blocked","transpose-blocked-32"]):
    
    data = pd.read_csv(source+"_mem.dat", delim_whitespace=True)
    data_tens = data[data["Nrow"] % 10 == 0]
    data_non_tens = data[data["Nrow"] % 10 != 0]

    for i, data in enumerate([data_tens, data_non_tens]):
        ax.plot(data["Nrow"], data["EffectiveBW"], marker='o', linestyle=linestyles[i], color=colors[j], label=f"{source} {'tens' if i==0 else '2^n'}")
        
ax.legend()
plt.show()
fig.savefig("mem-bench-plot.png", dpi=300)

