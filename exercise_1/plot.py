import matplotlib.pyplot as plt

sizes_bytes = []
mflops = []

def parse_size(size_str):
    if size_str.endswith("kB"):
        return float(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) * 1024 * 1024
    else:
        raise ValueError(f"Unknown size unit: {size_str}")

with open("benchmark.out", "r") as f:
    for line in f:
        if not line.strip():
            continue
        size_str, perf_str = line.split()
        sizes_bytes.append(parse_size(size_str))
        mflops.append(float(perf_str))

plt.figure()
plt.plot(sizes_bytes, mflops, marker="o")
plt.xscale("log")
plt.xlabel("Working set size (bytes)")
plt.ylabel("Performance (MFlops/s)")
plt.title("LIKWID sum_sp Performance vs Working Set Size")
plt.grid(True)

plt.savefig("benchmark.png", dpi=300, bbox_inches="tight")
plt.close()
