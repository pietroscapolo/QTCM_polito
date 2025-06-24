
# =============================================
# Numerical Study of the Interacting Bosonic SSH Model
# Author: Pietro Scapolo
# =============================================

# ----------------------------
# Imports and Setup
# ----------------------------
from quspin.basis import boson_basis_1d
from quspin.operators import hamiltonian
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Suppress warnings and stdout
stdout_backup = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Image saving directory
images_dir = os.path.join(os.getcwd(), "figures")
os.makedirs(images_dir, exist_ok=True)

# Image size
img_w, img_h = 3.6, 2.6

# ----------------------------
# Function to construct the SSH Hamiltonian
# ----------------------------
def bosonic_ssh_hamiltonian(*, L, J, deltaJ, U, Nbosons, sps, BC):
    if BC.upper() == "PBC":
        hop_list = [[- (J + (-1)**i * deltaJ), i, (i+1) % L] for i in range(L)]
    elif BC.upper() == "OBC":
        hop_list = [[- (J + (-1)**i * deltaJ), i, i+1] for i in range(L-1)]
    else:
        raise ValueError("BC must be 'PBC' or 'OBC'")

    nn_list = [[U/2, i, i] for i in range(L)]
    n_list = [[-U/2, i] for i in range(L)]

    static = [
        ["+-", hop_list],
        ["-+", hop_list],
        ["nn", nn_list],
        ["n", n_list],
    ]

    dynamic = []
    basis = boson_basis_1d(L=L, Nb=Nbosons, sps=sps)
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    return H, basis

# ----------------------------
# SECTION 1: Ground State Symmetry
# ----------------------------
def compute_ground_state_symmetry():
    for L, BC, tag in [(6, "OBC", "OBC_even"), (7, "OBC", "OBC_odd"),
                       (6, "PBC", "PBC_even"), (7, "PBC", "PBC_odd")]:
        N = L // 2
        deltaJs = np.linspace(-1, 1, 10)
        U = 2
        E_gs = {}

        for deltaJ in deltaJs:
            H, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N+1, sps=10, BC=BC)
            E, _ = H.eigh()
            E_gs[deltaJ] = E[0]

        plt.figure(figsize=(img_w, img_h))
        plt.plot([-dj for dj in deltaJs], [E_gs[dj] for dj in deltaJs], 'o-', color='#FF5A76',
                 label=r'$E_{\mathrm{gs}}(-\delta J)$', markersize=5, linewidth=1.5)
        plt.plot(deltaJs, [E_gs[dj] for dj in deltaJs], 'o-', color='#00C2A0',
                 label=r'$E_{\mathrm{gs}}(\delta J)$', markersize=5, linewidth=1.5)
        plt.xlabel(r'$\delta J$')
        plt.ylabel(r'$E_{\mathrm{gs}}$')
        plt.legend(frameon=False)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{images_dir}/Egs_{tag}.png", dpi=300)
        plt.show()

# ----------------------------
# SECTION 2: Charge Gap and Thermodynamic Limit
# ----------------------------
def compute_charge_gap(deltaJ, tag, order):
    import numpy as np
    import matplotlib.pyplot as plt

    L_list = [4, 6, 8]
    img_w, img_h = 3.6, 2.6
    color_palette = ['#00C2A0', '#FF5A76', '#0072B2']
    color_gap = '#FF5A76'  

    # U da usare per fit e per curva extrapolata (interpolati)
    if order == 2:
        U_list_fit = [1, 5, 10]
    else:
        U_list_fit = [1, 5, 10]  

    U_list_plot = np.linspace(0, 20, 20)

    # Dict per salvare i gap
    gaps = {}

    # ---- FASE 1: Calcolo dei gap per i U_list_fit (fit ∆c vs 1/L) ----
    for U in U_list_fit:
        gaps[U] = {}
        for L in L_list:
            N = L // 2
            H_plus, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N + 1, sps=N + 2, BC="PBC")
            H_minus, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N - 1, sps=N, BC="PBC")
            H, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N, sps=N + 1, BC="PBC")

            E_gs_plus = H_plus.eigvalsh()[0]
            E_gs_minus = H_minus.eigvalsh()[0]
            E_gs = H.eigvalsh()[0]

            gaps[U][L] = E_gs_plus + E_gs_minus - 2 * E_gs

    inv_L = np.array([1 / L for L in L_list])
    inv_L_fine = np.linspace(0, max(inv_L), 300)

    plt.figure(figsize=(img_w, img_h))
    for i, U in enumerate(U_list_fit):
        y = np.array([gaps[U][L] for L in L_list])
        coeffs = np.polyfit(inv_L, y, order)
        fit_fn = np.poly1d(coeffs)
        color = color_palette[i % len(color_palette)]
        plt.plot(inv_L, y, 'o', color=color, markersize=5)
        plt.plot(inv_L_fine, fit_fn(inv_L_fine), '--', color=color)
    plt.xlabel('1 / L')
    plt.ylabel(r'$\Delta_c$')
    plt.xlim(0, 0.26)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{images_dir}/fit_deltaj_{tag}.png", dpi=300)
    plt.show()

    # ---- FASE 2: Calcolo dei gap extrapolati a L to infty per molti U ----
    gap_at_infty = []

    for U in U_list_plot:
        if U not in gaps:
            gaps[U] = {}
            for L in L_list:
                N = L // 2
                H_plus, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N + 1, sps=N + 2, BC="PBC")
                H_minus, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N - 1, sps=N, BC="PBC")
                H, _ = bosonic_ssh_hamiltonian(L=L, J=1.0, deltaJ=deltaJ, U=U, Nbosons=N, sps=N + 1, BC="PBC")

                E_gs_plus = H_plus.eigvalsh()[0]
                E_gs_minus = H_minus.eigvalsh()[0]
                E_gs = H.eigvalsh()[0]

                gaps[U][L] = E_gs_plus + E_gs_minus - 2 * E_gs

        y = np.array([gaps[U][L] for L in L_list])
        coeffs = np.polyfit(inv_L, y, order)
        gap_at_infty.append(coeffs[-1])

    plt.figure(figsize=(img_w, img_h))
    plt.plot(U_list_plot, gap_at_infty, 'o-', color=color_gap, markersize=5)
    plt.xlabel('U')
    plt.ylabel(r'$\Delta_c(L \to \infty)$')
    if deltaJ == 0.2:
        plt.ylim(0, 0.4)
    else:
        plt.ylim(0, 1.8)
    plt.xlim(0, 20)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{images_dir}/phase_trans_deltaj_{tag}.png", dpi=300)
    plt.show()

# ----------------------------
# SECTION 3: Local Density Profile
# ----------------------------
def compute_local_density_profiles():
    L = 6                              # numero di siti
    N = (L // 2) + 1                   # bosoni: leggermente dopato
    sps = N + 1                        # max stati per sito (serve per la base)
    params = [(-0.9, 2), (-0.9, 30), (+0.9, 30)]  # (deltaJ, U)
    bar_colors = ['#00C2A0', '#FF5A76', '#0072B2']  # menta, rosa, blu
    labels = [
        r'$\delta J = -0.9,\ U = 2$',
        r'$\delta J = -0.9,\ U = 30$',
        r'$\delta J = +0.9,\ U = 30$'
    ]

    for i, (deltaJ, U) in enumerate(params):
        # Costruzione hamiltoniana e base per OBC
        H_obc, basis_obc = bosonic_ssh_hamiltonian(
            L=L, J=1.0, deltaJ=deltaJ, U=U,
            Nbosons=N, sps=sps, BC="OBC"
        )

        # Stato fondamentale
        E_obc, V_obc = H_obc.eigh()
        gs_obc = V_obc[:, 0]

        # Calcolo della densità locale media normalizzata
        n_avg_obc = np.array([
            hamiltonian([["n", [[1.0, j]]]], [], basis=basis_obc).expt_value(gs_obc)
            for j in range(L)
        ]) / N

        # Plotting
        x = np.arange(1, L + 1)
        plt.figure(figsize=(img_w, img_h / 1.5))
        plt.bar(x, n_avg_obc, color=bar_colors[i])
        plt.ylim(0, 0.35)
        plt.ylabel(r'$\langle n_i \rangle / N$', fontsize=12)
        plt.xlabel('Site index $i$', fontsize=12)
        plt.xticks(x, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(labels[i], fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{images_dir}/avg_boson_num_{i}.png", dpi=300)
        plt.show()

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    compute_ground_state_symmetry()
    compute_charge_gap(0.2, "02", 2)
    compute_charge_gap(0.6, "06", 1)
    compute_local_density_profiles()
