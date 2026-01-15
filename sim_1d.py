import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from typing import Dict, Any, Tuple, List

# ============================================================
# 1. CORE KURAMOTO–HOMOPHILY DYNAMICS (1D angles)
# ============================================================

def kuramoto_step(theta, omega, G, g, h, dt=0.01):
    """
    Euler step of Kuramoto-homophily dynamics:
        dθ_i = ω_i + (g/k_i) * Σ_j [ sin(θ_j-θ_i) - h sin(2(θ_j-θ_i)) ]
    """
    N = len(theta)
    dtheta = np.zeros(N)
    for i in range(N):
        neighbors = list(G.neighbors(i))
        if len(neighbors) == 0:
            dtheta[i] = omega[i]
            continue
        k_i = len(neighbors)
        influence = 0.0
        for j in neighbors:
            diff = theta[j] - theta[i]
            influence += np.sin(diff) - h * np.sin(2 * diff)
        dtheta[i] = omega[i] + (g / k_i) * influence
    return theta + dt * dtheta

# ============================================================
# 2. METRICS AND DIAGNOSTICS
# ============================================================

def order_parameter(theta):
    """Global coordination R = |mean of unit complex| in [0,1]."""
    return np.abs(np.mean(np.exp(1j * theta)))

def compute_clusters_1d(theta, max_k=8):
    """
    Cluster detection on circle via histogram peak finding.
    Returns: n_clusters, centers (angles), assign (indices), within variances, min separation
    """
    angles = np.mod(theta, 2*np.pi)
    hist, bins = np.histogram(angles, bins=36, range=(0, 2*np.pi))
    peaks, _ = find_peaks(hist, distance=3, height=np.mean(hist))
    n_clusters = max(2, min(len(peaks), max_k))
    # centers from peaks
    if len(peaks) == 0:
        centers = np.linspace(0, 2*np.pi, num=n_clusters, endpoint=False)
    else:
        centers = (bins[peaks] + bins[peaks + 1]) / 2
        if len(centers) < n_clusters:
            extra = np.linspace(0, 2*np.pi, num=n_clusters - len(centers), endpoint=False)
            centers = np.concatenate([centers, extra])
    assign = np.argmin(np.abs(angles[:, None] - centers[None, :]), axis=1)
    variances = []
    center_list = []
    for c in range(n_clusters):
        pts = angles[assign == c]
        if len(pts) >= 4:
            variances.append(np.var(pts))
            center_list.append(np.mean(pts))
    if len(center_list) >= 2:
        sorted_centers = np.sort(np.array(center_list))
        diffs = np.diff(sorted_centers)
        wrap = 2*np.pi - (sorted_centers[-1] - sorted_centers[0])
        diffs = np.append(diffs, wrap)
        sep = np.min(diffs[diffs > 0.2]) if np.any(diffs > 0.2) else np.nan
    else:
        sep = np.nan
    return n_clusters, np.array(center_list), assign, np.array(variances), sep

def compute_VDR_inv(theta):
    """
    VDR^{-1} = mean within-cluster variance / separation of cluster means (angular).
    """
    n_clusters, centers, assign, variances, sep = compute_clusters_1d(theta)
    if len(variances) < 2 or np.isnan(sep) or sep < 1e-6:
        return np.nan
    within = np.mean(variances)
    return within / sep

def distribution_moments(theta):
    """
    Skewness and kurtosis of angles mapped to [-pi, pi] and projected on sin/cos to avoid wrap artifacts.
    """
    # Use components to avoid circular discontinuities
    x = np.cos(theta)
    y = np.sin(theta)
    # Moments over components (captures shape without wrap issues)
    sk = 0.5 * (skew(x) + skew(y))
    ku = 0.5 * (kurtosis(x, fisher=True) + kurtosis(y, fisher=True))
    return sk, ku

def susceptibility_proxy(G, g, h, X_unit=None):
    """
    Spectral proxy for χ(t):
    χ ~ g * λ_max(A) * L_local, where L_local encodes local linearization gains from F'.
    For 1D, F'(Δ) ≈ cos(Δ) - 2h cos(2Δ). Approximate L_local by mean of |cos| + 2h|cos(2)|.
    If X_unit provided (angles), compute empirical mean; else assume moderate value (~0.5 + h).
    """
    # Largest eigenvalue of adjacency
    A = nx.to_numpy_array(G)
    lam_max = np.linalg.eigvalsh(A).max()
    if X_unit is None:
        L_local = 0.5 + h  # conservative proxy
    else:
        ang = X_unit
        L_local = np.mean(np.abs(np.cos(ang)) + 2*h*np.abs(np.cos(2*ang)))
    return g * lam_max * L_local

def detect_onset(series, smooth=3, dthr=0.01):
    """
    Detect onset of rapid increase using derivative threshold on smoothed series.
    Returns first index where d/dt exceeds dthr.
    """
    if len(series) < smooth+1:
        return np.nan
    s = np.convolve(series, np.ones(smooth)/smooth, mode='valid')
    ds = np.diff(s)
    idx = np.where(ds > dthr)[0]
    return (idx[0] + (smooth-1)) if len(idx) > 0 else np.nan

def lead_time(vdr, R, min_points=10):
    """
    Estimate lead time between VDR^{-1} spike onset and R(t) onset.
    """
    if len(vdr) < min_points or len(R) < min_points:
        return np.nan
    v_on = detect_onset(vdr, smooth=5, dthr=np.nanmean(np.diff(vdr)) + np.nanstd(np.diff(vdr)))
    r_on = detect_onset(R, smooth=5, dthr=np.nanmean(np.diff(R)) + np.nanstd(np.diff(R)))
    if np.isnan(v_on) or np.isnan(r_on):
        return np.nan
    return r_on - v_on

# ============================================================
# 3. MACRO SIMULATION (SCALE-FREE NETWORK) WITH DIAGNOSTICS
# ============================================================

def macro_simulation(
        N=2000, g=1.4, h=0.9, T=2000, dt=0.01, sample_every=50,
        plot=True, seed=1):

    rng = default_rng(seed)
    G = nx.barabasi_albert_graph(N, 3)

    theta = rng.uniform(0, 2*np.pi, N)
    omega = rng.normal(0, 0.2, N)

    R_values, VDR_inv_values = [], []
    skew_vals, kurt_vals, chi_vals = [], [], []
    snapshots = []

    for t in range(T):
        theta = kuramoto_step(theta, omega, G, g, h, dt)
        if t % sample_every == 0:
            R_values.append(order_parameter(theta))
            VDR_inv_values.append(compute_VDR_inv(theta))
            sk, ku = distribution_moments(theta)
            skew_vals.append(sk); kurt_vals.append(ku)
            chi_vals.append(susceptibility_proxy(G, g, h, X_unit=theta))
            snapshots.append(theta.copy())

    lt = lead_time(np.array(VDR_inv_values, dtype=float), np.array(R_values, dtype=float))

    if plot:
        fig, axs = plt.subplots(5,1, figsize=(10,18), sharex=True)
        axs[0].plot(R_values); axs[0].set_title("MACRO 1D: Order Parameter R(t)")
        axs[0].set_ylabel("R")
        axs[1].plot(VDR_inv_values); axs[1].set_title("MACRO 1D: VDR^{-1}(t)")
        axs[1].set_ylabel("VDR^{-1}")
        axs[2].plot(chi_vals); axs[2].set_title("Susceptibility proxy χ(t)")
        axs[2].set_ylabel("χ")
        axs[3].plot(skew_vals, label="skew"); axs[3].plot(kurt_vals, label="kurtosis")
        axs[3].set_title("Distribution moments over time"); axs[3].legend()
        axs[4].hist(theta, bins=60, density=True, color='gray', alpha=0.8)
        axs[4].set_title("Final phase distribution (macro 1D)")
        axs[4].set_xlabel("time / {}".format(sample_every))
        plt.tight_layout()
        plt.savefig("macro_1d_enhanced.png")
        plt.show()
        if not np.isnan(lt):
            print(f"Estimated VDR lead over R onset (in samples): {lt}")

    return theta, R_values, VDR_inv_values, chi_vals, skew_vals, kurt_vals, snapshots, lt

# ============================================================
# 4. MICRO SIMULATION (Small-World Communities) WITH SEEDS
# ============================================================

def micro_simulation(
        num_communities=5, N_per=150, k=6, p=0.05,
        g=0.6, h=0.3, T=1500, dt=0.02, sample_every=50,
        plot=True, seed=2, seeds_per_community=3):

    rng = default_rng(seed)
    results = []

    for c in range(num_communities):
        # Fixed topology per community
        G = nx.watts_strogatz_graph(N_per, k, p)
        community_runs = []
        for s in range(seeds_per_community):
            theta = rng.uniform(0, 2*np.pi, N_per)
            omega = rng.normal(0, 0.25, N_per)
            R_values, VDR_inv_values = [], []
            for t in range(T):
                theta = kuramoto_step(theta, omega, G, g, h, dt)
                if t % sample_every == 0:
                    R_values.append(order_parameter(theta))
                    VDR_inv_values.append(compute_VDR_inv(theta))
            community_runs.append((np.array(R_values), np.array(VDR_inv_values), theta.copy()))
        results.append((G, community_runs))

    if plot:
        fig, axs = plt.subplots(num_communities, 3, figsize=(15, 4*num_communities))
        for c in range(num_communities):
            runs = results[c][1]
            R_mat = np.stack([r[0] for r in runs], axis=0)
            V_mat = np.stack([r[1] for r in runs], axis=0)
            R_mean = np.nanmean(R_mat, axis=0)
            R_std = np.nanstd(R_mat, axis=0)
            V_mean = np.nanmean(V_mat, axis=0)
            V_std = np.nanstd(V_mat, axis=0)
            x = np.arange(R_mean.shape[0])

            axs[c,0].plot(x, R_mean, color='C0')
            axs[c,0].fill_between(x, R_mean-R_std, R_mean+R_std, alpha=0.2, color='C0')
            axs[c,0].set_title(f"COMMUNITY {c}: R(t) mean ± std")

            axs[c,1].plot(x, V_mean, color='C1')
            axs[c,1].fill_between(x, V_mean-V_std, V_mean+V_std, alpha=0.2, color='C1')
            axs[c,1].set_title(f"COMMUNITY {c}: VDR^{-1}(t) mean ± std")

            final_theta = runs[-1][2]
            axs[c,2].hist(final_theta, bins=50, density=True)
            axs[c,2].set_title(f"COMMUNITY {c}: Final distribution (last seed)")

        plt.tight_layout()
        plt.savefig("micro_1d_enhanced.png")
        plt.show()

    return results

# ============================================================
# 5. MULTI-SCALE RUN
# ============================================================

if __name__ == "__main__":
    print("Running macro-level 1D enhanced simulation...")
    macro_out = macro_simulation(
        N=2000, g=1.4, h=0.9, T=2500, dt=0.01, sample_every=50, plot=True, seed=42
    )

    print("Running micro-level 1D communities with robustness...")
    micro_out = micro_simulation(
        num_communities=5, N_per=150, g=0.6, h=0.3,
        T=1500, dt=0.02, sample_every=50, plot=True, seed=123, seeds_per_community=5
    )
