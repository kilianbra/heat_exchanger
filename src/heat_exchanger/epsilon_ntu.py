import numpy as np
from matplotlib import pyplot as plt
from scipy.special import factorial


def poly_sum_crossflow_unmixed(n, y):
    y = np.asarray(y)
    result = np.empty_like(y, dtype=float)

    # Mask for y < 0: set to np.nan
    mask_neg = y < 0
    result[mask_neg] = np.nan

    # Mask for 0 <= y < 1e-8: set to 0
    mask_small = (y >= 0) & (y < 1e-8)
    result[mask_small] = 0.0

    # Mask for y >= 1e-8: compute as before
    mask_valid = y >= 1e-8
    if np.any(mask_valid):
        y_valid = y[mask_valid]
        sum_term = np.sum(
            [(n + 1 - j) / factorial(j) * np.exp((j + n) * np.log(y_valid)) for j in range(1, n + 1)],
            axis=0,
        )
        result[mask_valid] = (1.0 / factorial(n + 1)) * sum_term

    return result


def epsilon_ntu(NTU, C_ratio, exchanger_type="aligned_flow", flow_type="counterflow", n_passes=1):
    """
    Calculate the effectiveness (epsilon) of a heat exchanger using the Number of Transfer Units (NTU) method.

    Parameters:
    NTU (float/np array): Number of Transfer Units
    C_ratio (float): Capacity rate ratio 0 <= C_min/C_max <= 1
    exchanger_type (str, optional): Type of the heat exchanger. Can be 'aligned_flow', 'cross_flow', or 'shell_and_tube'. Default is 'aligned_flow'.
    flow_type (str, optional): Arrangement of the flow. Can be 'parallel', 'counter', 'unmixed', 'Cmax_mixed', 'Cmin_mixed', or 'both_mixed'. Default is 'counterflow'.
    n_passes (int, optional): Number of passes (overall counterflow for all types, except for shell in tube where this is shell passes). Default is 1.

    Returns:
    epsilon (float): Effectiveness of the heat exchanger
    """
    assert 0 <= C_ratio <= 1, f"C_ratio should be between 0 and 1 (inclusive), but is {C_ratio:.1f}"
    assert isinstance(n_passes, int) and n_passes >= 1, (
        f"n_passes should be an integer greater than or equal to 1 but is {n_passes:.0f}"
    )
    valid_exchanger_types = ["aligned_flow", "cross_flow", "shell_and_tube"]
    valid_flow_types = [
        "coflow",
        "counterflow",
        "unmixed",
        "Cmax_mixed",
        "Cmin_mixed",
        "both_mixed",
    ]
    assert exchanger_type in valid_exchanger_types, f"Invalid exchanger_type. Must be one of {valid_exchanger_types}"
    assert flow_type in valid_flow_types, f"Invalid flow_type. Must be one of {valid_flow_types}"

    tol = 1e-9
    tol_it = 1e-4
    ntu_p = NTU / n_passes
    if C_ratio < 0 + tol:  # Close enough to zero, doesn't matter the type as C_min fluid doesn't change temperature
        epsilon = 1 - np.exp(-ntu_p)  # Kays & London (2-13a)

    elif exchanger_type == "aligned_flow":  # Two concentric tubes aligned with eachother
        if flow_type == "coflow":
            epsilon = (1 - np.exp(-ntu_p * (1 + C_ratio))) / (1 + C_ratio)  # Kays & London (2-14)
        elif flow_type == "counterflow":
            if C_ratio < 1 - tol:  # Far enough from 1
                epsilon = (1 - np.exp(-ntu_p * (1 - C_ratio))) / (
                    1 - C_ratio * np.exp(-ntu_p * (1 - C_ratio))
                )  # Kays & London (2-13)
            else:  # Close enough to 1
                epsilon = ntu_p / (ntu_p + 1)  # Kays & London (2-13b)
        if (
            n_passes > 1
        ):  # we have just calculated the effectiveness for one pass, assumes mixing between passes and overall counterflow config
            epsilon_1p = epsilon
            if C_ratio > 1 - tol:  # Close enough to 1 that will be considered as such
                epsilon = n_passes * epsilon_1p / (1 + (n_passes - 1) * epsilon_1p)  # Kays & London (2-18a)
            elif n_passes == 1:
                epsilon = epsilon_1p
            else:  # Kays & London (2-18)
                epsilon = (((1 - epsilon_1p * C_ratio) / (1 - epsilon_1p)) ** n_passes - 1) / (
                    (((1 - epsilon_1p * C_ratio) / (1 - epsilon_1p)) ** n_passes) - C_ratio
                )

    elif exchanger_type == "cross_flow":
        if flow_type == "unmixed":
            # This is a computationally efficient way to estimate, but is just an approximation
            # epsilon = 1 - np.exp(ntu_p**0.22 / C_ratio * (np.exp(-C_ratio * ntu_p**0.78) - 1)) #CHECK REFERENCE Eqn (3.24) in Lopez -> This is a decent approximation but doesn't asymptote to right answer, which is below
            # Shah p128 Table 3.3 formula for crossflow, both fluids unmixed
            epsilon = 1 - np.exp(-NTU)
            n = 1
            term = -np.exp(-(1 + C_ratio) * NTU) * (C_ratio**n) * poly_sum_crossflow_unmixed(n, NTU)
            # Loop until the added term is smaller than the tolerance
            while np.any(np.abs(term) > tol_it):
                epsilon += term
                n += 1
                term = -np.exp(-(1 + C_ratio) * NTU) * (C_ratio**n) * poly_sum_crossflow_unmixed(n, NTU)
            return epsilon

        elif flow_type == "Cmax_mixed":
            # epsilon = 1 / C_ratio * (1 - np.exp(1 - C_ratio) * (1 - np.exp(-ntu_p))) # Lopez (3.25) WRONG
            epsilon = 1 / C_ratio * (1 - np.exp(-C_ratio * (1 - np.exp(-ntu_p))))  # Kays & London (2-16)
        elif flow_type == "Cmin_mixed":
            epsilon = 1 - np.exp(-1 / C_ratio * (1 - np.exp(-C_ratio * ntu_p)))  # Kays & London (2-15)
        elif flow_type == "both_mixed":
            epsilon = 1 / (
                1 / (1 - np.exp(-ntu_p)) + C_ratio / (1 - np.exp(-C_ratio * ntu_p)) - 1 / ntu_p
            )  # Kays & London (2-17)

        if (
            n_passes > 1
        ):  # we have just calculated the effectiveness for one pass, assumes mixing between passes and overall counterflow config
            epsilon_1p = epsilon
            if C_ratio > 1 - tol:  # Close enough to 1 that will be considered as such
                epsilon = n_passes * epsilon_1p / (1 + (n_passes - 1) * epsilon_1p)  # Kays & London (2-18a)
            elif n_passes == 1:
                epsilon = epsilon_1p
            else:  # Kays & London (2-18)
                epsilon = (((1 - epsilon_1p * C_ratio) / (1 - epsilon_1p)) ** n_passes - 1) / (
                    (((1 - epsilon_1p * C_ratio) / (1 - epsilon_1p)) ** n_passes) - C_ratio
                )

    elif exchanger_type == "shell_and_tube":
        ntu_p = NTU / n_passes  # Number of heat transfer units in case of shell-and-tube exchanger
        # 1 pass epsilon, from Kays & London (2-19) Assumes  mixing  between passes, but Kays & London says this doesn't affect much
        C_ratio_sqrt = np.sqrt(1 + C_ratio**2)
        epsilon_1p = np.where(
            ntu_p < 10,
            2
            / (1 + C_ratio + C_ratio_sqrt * (1 + np.exp(-ntu_p * C_ratio_sqrt)) / (1 - np.exp(-ntu_p * C_ratio_sqrt))),
            2 / (1 + C_ratio + C_ratio_sqrt),
        )
        if C_ratio > 1 - tol:  # Close enough to 1 that will be considered as such
            epsilon = (
                n_passes * epsilon_1p / (1 + (n_passes - 1) * epsilon_1p)
            )  # Kays & London (2-18a) works for all  shell_pass values
        elif n_passes == 1:
            epsilon = epsilon_1p
        else:  # Kays & London (2-18)
            epsilon = (((1 - epsilon_1p * C_ratio) / (1 - epsilon_1p)) ** n_passes - 1) / (
                (((1 - epsilon_1p * C_ratio) / (1 - epsilon_1p)) ** n_passes) - C_ratio
            )

    return epsilon


if __name__ == "__main__":
    if True:  # C_ratio variation
        HX_tp = "cross_flow"
        flow_tp = "unmixed"
        NTUs = np.linspace(0, 50, 500)
        C_ratios = [0, 0.25, 0.5, 0.75, 0.9, 1]
        for C_ratio in C_ratios:
            eps = []
            # for NTU in NTUs:
            eps = epsilon_ntu(NTUs, C_ratio, exchanger_type=HX_tp, flow_type=flow_tp, n_passes=1)
            # eps = epsilon_ntu(NTUs,C_ratio,exchanger_type=HX_tp, flow_type='unmixed', n_passes=1)
            plt.plot(NTUs, eps, label=rf"$C_r$ = {C_ratio:.2f}")
        # Set font properties
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 12
        plt.xlabel("NTU")
        plt.ylabel(r"$\varepsilon$")
        plt.title(f" HX ({flow_tp}, {HX_tp}) heat transfer")
        plt.legend()

        plt.xlim(0, 7)
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True, which="both")
        # filename = f"PhD_scripts/Figs_PhD/eps_NTU_{HX_tp}_{flow_tp}.svg"
        # plt.savefig(filename, format="svg")
        plt.show()
    else:  # number of pass variation
        HX_tp = "cross_flow"
        flow_tp = "Cmin_mixed"
        NTUs = np.linspace(0, 50, 500)
        n_passes = [1, 2, 3, 4, 6, 999]
        C_ratio = 1
        for n_pass in n_passes:
            eps = []
            # for NTU in NTUs:
            eps = epsilon_ntu(NTUs, C_ratio, exchanger_type=HX_tp, flow_type=flow_tp, n_passes=n_pass)

            if n_pass > 10:
                n_pass = r"$\infty$"
            plt.plot(NTUs, eps, label=rf"$n_p$ = {n_pass}")
        plt.xlabel("NTU")
        plt.ylabel(r"$\varepsilon$")
        plt.title(rf" HX ({flow_tp}, {HX_tp}, $C_r$={C_ratio}) heat transfer")
        plt.legend()

        plt.xlim(0, 7)
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True, which="both")
        # filename = f"PhD_scripts/Figs_PhD/eps_NTU_{HX_tp}_{flow_tp}_multipass.svg"
        # plt.savefig(filename,format='svg')
        plt.show()

    """
    #Test values
    C_min = 0.063 *  14500
    C_ratio = 0.063 *  14500  /  10 / 1130
    exchanger_type = 'shell_and_tube'
    n_passes = 1
    ls = ["k--","k-", "k-."]
    for i,C  in  enumerate([0,C_ratio,1]):
        NTU  = np.linspace(0.1,20,100)
        epsilon = epsilon_ntu(NTU, C, exchanger_type, n_passes=n_passes)
        #plt.plot(NTU, epsilon, ls[i], label=rf'$C_r$ = {C:.2f}')
        if C==C_ratio:
            eps_max = 0.99*epsilon[-1]
            indices = np.where(epsilon < eps_max)
            if indices[0].size > 0:
                largest_NTU = NTU[indices[-1][-1]]
            else:
                largest_NTU = None
            #print(f"The highest NTU which is 1% lower than eps_max = {epsilon[-1]:.2%} is NTU={largest_NTU:.2f}")


    #Rating on one HX (low dp)
    epsilon = [0.7,0.75,0.8, 0.85, 0.9, 0.92, 0.94,0.95,0.952]
    U_A_req = np.array([18.3*64.9,20.7*67.2,23.9*69,28.6*70.3, 37.1*70.6, 43.3*70.1, 54.9*69.5, 74*69.3, 93.5*68.4])
    oversize_fact = np.array([14.59,12.9,11.16,9.3, 7.18, 6.16, 4.86, 3.6, 2.85])
    dp = np.array([0.08807,0.08779,0.08747,0.08718,0.0873,0.08767,0.08834,0.0884])/1.5
    U_A_max = U_A_req * oversize_fact

    print(f"Average pressure drop: {np.mean(dp):.2%} +/- {np.std(dp)*100:.2f} of p_in")

    #plt.scatter(U_A_req/C_min, epsilon,label = "Aspen req. area",marker="+")
    plt.scatter(U_A_max/C_min, epsilon,label = "Aspen act. area",marker="x")
    plt.grid(True)
    plt.xlabel(r'Number of Transfer Units  $ N_{TU}  = UA/C_{min} (-)$')
    plt.ylabel(r'Effectiveness $\varepsilon =  q/q_{max}  (-)$')
    plt.legend()
    #plt.scatter([385.7],[0.986])
    plt.title('Single pass Shell & Tube HX effectiveness curve')
    plt.savefig('plots/eps_NTU_with_Aspen.svg', format='svg')
    plt.show()

    """
