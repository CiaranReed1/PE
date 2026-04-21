import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data
serial_data = pd.read_csv("serial_timings.csv")
cuda_kernel_data = pd.read_csv("cuda_kernel_timings.csv")
cuda_whole_data = pd.read_csv("cuda_timings.csv")
cuda_tasks_data = pd.read_csv("cuda_tasks_timings.csv")

functions = ["t_a", "t_b", "t_c", "t_d", "t_e"]
plot_functions = ["t_b", "t_c", "t_d", "t_e"]
Ns = [1000,5000,10000,15000,20000,25000,30000]

# ---- First plot: serial means with standard error ----
serial_mean = serial_data.groupby("N")[functions].mean()
serial_sem  = serial_data.groupby("N")[functions].sem()

x = np.arange(len(functions))

fig, ax = plt.subplots(figsize=(10, 6))

for N in Ns:
    ax.errorbar(
        x,
        serial_mean.loc[N, functions],
        yerr=serial_sem.loc[N, functions],
        fmt='o',
        capsize=5,
        label=f"N = {N}"
    )

ax.set_xticks(x)
ax.set_xticklabels(functions)
ax.set_xlabel("Function")
ax.set_ylabel("Time (s)")
ax.set_title("Serial execution time with standard error")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/serial_timings.png")


# ---- Compute means and SEM for all datasets ----
serial_mean = serial_data.groupby("N")[functions].mean()
serial_sem  = serial_data.groupby("N")[functions].sem()

cuda_kernel_mean = cuda_kernel_data.groupby("N")[functions].mean()
cuda_kernel_sem  = cuda_kernel_data.groupby("N")[functions].sem()

cuda_whole_mean = cuda_whole_data.groupby("N")[functions].mean()
cuda_whole_sem  = cuda_whole_data.groupby("N")[functions].sem()


# ---- One plot per function (excluding t_a) ----
for func in plot_functions:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        Ns,
        serial_mean.loc[Ns, func],
        yerr=serial_sem.loc[Ns, func],
        fmt='-o',
        capsize=5,
        label="Serial"
    )

    ax.errorbar(
        Ns,
        cuda_kernel_mean.loc[Ns, func],
        yerr=cuda_kernel_sem.loc[Ns, func],
        fmt='-o',
        capsize=5,
        label="CUDA kernel only"
    )

    ax.errorbar(
        Ns,
        cuda_whole_mean.loc[Ns, func],
        yerr=cuda_whole_sem.loc[Ns, func],
        fmt='-o',
        capsize=5,
        label="CUDA total"
    )

    ax.set_xlabel("Problem size (N)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{func} duration vs problem size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{func}_timings.png")


# ---- Total runtime comparison: Serial vs CUDA tasks ----
serial_total_mean = serial_data.groupby("N")["total_time"].mean()
serial_total_sem  = serial_data.groupby("N")["total_time"].sem()

cuda_tasks_total_mean = cuda_tasks_data.groupby("N")["total_time"].mean()
cuda_tasks_total_sem  = cuda_tasks_data.groupby("N")["total_time"].sem()

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    Ns,
    serial_total_mean.loc[Ns],
    yerr=serial_total_sem.loc[Ns],
    fmt='-o',
    capsize=5,
    label="Serial"
)

ax.errorbar(
    Ns,
    cuda_tasks_total_mean.loc[Ns],
    yerr=cuda_tasks_total_sem.loc[Ns],
    fmt='-o',
    capsize=5,
    label="CUDA asynchronous tasks"
)

ax.set_xlabel("Problem size (N)")
ax.set_ylabel("Total time (s)")
ax.set_title("Total runtime: Serial vs CUDA asynchronous tasks")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/total_time_serial_vs_tasks.png")

# ---- Quadratic regression for total runtime ----

# Extract means
serial_y = serial_total_mean.loc[Ns].values
cuda_tasks_y = cuda_tasks_total_mean.loc[Ns].values

# Perform quadratic regression (degree 2 polynomial)
serial_coeffs = np.polyfit(Ns, serial_y, 2)
cuda_coeffs = np.polyfit(Ns, cuda_tasks_y, 2)

# Extract coefficients
a_serial, b_serial, c_serial = serial_coeffs
a_cuda, b_cuda, c_cuda = cuda_coeffs

# Generate smooth N values for plotting fitted curves
N_fit = np.linspace(min(Ns), max(Ns), 200)

serial_fit = a_serial * N_fit**2 + b_serial * N_fit + c_serial
cuda_fit = a_cuda * N_fit**2 + b_cuda * N_fit + c_cuda

# ---- Plot with quadratic regression curves ----
fig, ax = plt.subplots(figsize=(8, 5))

# Original data with error bars
ax.errorbar(
    Ns,
    serial_y,
    yerr=serial_total_sem.loc[Ns],
    fmt='o',
    capsize=5,
    label="Serial (data)"
)

ax.errorbar(
    Ns,
    cuda_tasks_y,
    yerr=cuda_tasks_total_sem.loc[Ns],
    fmt='o',
    capsize=5,
    label="CUDA tasks (data)"
)

# Fitted curves
ax.plot(
    N_fit,
    serial_fit,
    '--',
    label=f"Serial fit: T(N) = {a_serial:.2e}N² + {b_serial:.2e}N + {c_serial:.2e}"
)

ax.plot(
    N_fit,
    cuda_fit,
    '--',
    label=f"CUDA tasks fit: T(N) = {a_cuda:.2e}N² + {b_cuda:.2e}N + {c_cuda:.2e}"
)

ax.set_xlabel("Problem size (N)")
ax.set_ylabel("Total time (s)")
ax.set_title("Total runtime with quadratic regression")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/total_time_quadratic_regression.png")