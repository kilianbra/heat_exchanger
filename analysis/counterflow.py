import numpy as np

from heat_exchanger.epsilon_ntu import epsilon_ntu

if __name__ == "__main__":
    ntu = np.linspace(0, 2, 10)
    c_ratio = 0.871
    eps = epsilon_ntu(ntu, c_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1)

    # Print as a two-column table: NTU \t epsilon
    print("NTU\tepsilon")
    for n, e in zip(ntu, eps, strict=True):
        print(f"{n:.6f}\t{e:.6f}")
