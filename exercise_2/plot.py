import matplotlib.pyplot as plt

sizes_bytes = {}
bandwidths = {}

def parse_size(size_str):
    if size_str.endswith("kB"):
        return float(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return float(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Unknown size unit: {size_str}")

colors = ["blue", "orange", "green"]
files = ["clcopy","clload","clstore"]
for file in files:
    sizes_bytes[file] = []
    bandwidths[file] = []
    filename = f"{file}.dat"
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            size_str, bw_str = line.split()
            size = parse_size(size_str)
            bandwidth = float(bw_str)
            sizes_bytes[file].append(size)
            bandwidths[file].append(bandwidth)


plt.figure()
for i,file in enumerate(files):
    plt.plot(sizes_bytes[file], bandwidths[file], marker="o",color= colors[i], label=file)
plt.xscale("log")
plt.xlabel("Working set size (bytes)")
plt.ylabel("Bandwidth (MB/s)")
plt.title("LIKWID benchmarks Bandwidth vs Working Set Size")
plt.vlines([32*1024, 512*1024, 32*1024*1024], colors=["brown","brown","brown"],ymin=0, ymax=max(max(bandwidths[file]) for file in files), linestyles="dashed",label="Cache Sizes")
plt.legend()
plt.savefig("benchmark.png", dpi=300, bbox_inches="tight")
plt.close()
