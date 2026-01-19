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
ax.loglog(AI_range, np.minimum(compute_roof, memory_bandwidth*AI_range), label='Roofline Model', color='gray')
ax.scatter(const_AI,roofline_data["flops"]*1000000, label='Measured Performance', color='red',marker= "x")
ax.legend()
ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
ax.set_ylabel('Performance (FLOP/s)')
ax.set_title('Roofline Plot (O0 Optimization)')
plt.savefig('roofline_plot_O0.png')
plt.show()