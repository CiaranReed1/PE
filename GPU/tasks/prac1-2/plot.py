import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

fig,axs = plt.subplots(1,1,figsize=(15,5))
data = pd.read_csv("results.dat",delim_whitespace=True)
print(data)
tasks = ["task3","task4-a","task4-b"]
colors = ["blue","orange","green"]
for i,task in enumerate(tasks):
    linear_model = stats.linregress(data["N"],data[task])
    axs.plot(data["N"],linear_model.intercept + linear_model.slope*data["N"],color=colors[i],linestyle="--")
    label = f"{task} (slope={linear_model.slope:.2e}), intercept={linear_model.intercept:.2e}"
    axs.plot(data["N"],data[task],label=label,color=colors[i],marker = "x",linestyle="") 
axs.set_xlabel("N")
axs.set_ylabel("Time (s)")
axs.legend()
axs.set_title("Timing of Tasks 3 and 4")
plt.savefig("timing.png")  