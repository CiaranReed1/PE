import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

files = ["L1.dat", "L2.dat", "L3.dat", "MEM.dat"]

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.set_xlabel("N Cores")
ax.set_ylabel("Bandwidth [MB/s]")
ax.set_title("Memory Bandwidth Scaling with Number of Cores for different cache levels")
ax.set_yscale("log")
ax.set_xscale("log", base=2)
for file in files:
    data = pd.read_csv(file, delim_whitespace=True, header=None, names=["Cores","Bandwidth"])
    level = file.split(".")[0]
    ax.plot(data["Cores"], data["Bandwidth"], marker='x', label=level)
ax.legend()
plt.savefig("memory_bandwidth_scaling.png", dpi=300, bbox_inches="tight")
plt.close()