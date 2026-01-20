import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
roofline_data = pd.read_csv('roofline.out',names=['iter','rows','cols','flops'],header=None,delim_whitespace=True)
print(roofline_data.head())
memory_bandwidth = 19017.15e6 # B/s
compute_roof = 53.6e9 # FLOP/s
AI = 0.25 # FLOP/byte
exptected_peak = min(compute_roof,memory_bandwidth*AI)
AI_range = np.logspace(-2,2,100)
const_AI = roofline_data['flops']*0 + AI
fig, ax = plt.subplots(1,1,figsize=(10,6))
blocked = ["","-blocked"]
optimisation_levels = ["O0","O1", "O2", "O3"]
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
i = 0
for block in blocked:
    for level in optimisation_levels:
        roofline_data = pd.read_csv(f'roofline{block}_{level}.dat',delim_whitespace=True)
        const_AI = roofline_data['flops']*0 + AI
        ax.scatter(const_AI,roofline_data["flops"]*1000000, label=f'Measured Performance {level} {block}', color=colors[i],marker= "o" if block=="" else "x")
        i += 1
    
ax.loglog(AI_range, np.minimum(compute_roof, memory_bandwidth*AI_range), label='Roofline Model', color='gray')
ax.legend()
ax.set_xlim(1e-1, 1e0)
ax.set_ylim(1e8, 1e10)
ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax.set_ylabel('Performance (FLOP/s)')
ax.set_title('Roofline Plot (O0 Optimization)')
plt.savefig('roofline_plot_O0.png')
plt.show()