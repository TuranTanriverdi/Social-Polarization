"""
2D Kuramoto-homophily simulation with unified diagnostics.
Now uses metrics.py for all computations to ensure consistency.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import default_rng
import metrics  # Our unified diagnostics module


# ============================================================
# 1. CORE KURAMOTO-HOMOPHILY DYNAMICS → 2D UNIT VECTORS
# ============================================================

def kuramoto_step_2d(X, omega, G, g, h, dt=0.01):
    """
    2D Kuramoto-homophily on unit circle.
    X: (N,2) array of unit vectors
    omega: (N,) intrinsic angular velocities
    
    Dynamics:
        tangent attraction ∝ sin(Δθ)
        homophily/frustration ∝ -h·sin(2Δθ)
    """
    N = X.shape[0]
    dX = np.zeros_like(X)
    R90 = np.array([[0, -1], [1, 0]])  # 90° rotation

    for i in range(N):
        neighbors = list(G.neighbors(i))
        xi = X[i]
        xi_perp = R90 @ xi

        if len(neighbors) == 0:
            dX[i] = omega[i] * xi_perp
            continue

        k_i = len(neighbors)
        attraction = np.zeros(2)
        homophily = np.zeros(2)

        for j in neighbors:
            xj = X[j]
            diff = xj - xi
            dot = np.dot(xi, xj)
            norm_diff = np.linalg.norm(diff)
            
            # Tangent projection ∝ sin(Δθ)
            sin_dtheta = np.dot(xj, xi_perp)
            attraction += norm_diff * sin_dtheta * xi_perp
            
            # Homophily: -h·sin(2Δθ) ≈ -h·cos(Δθ)·(xj - xi)
            homophily += dot * diff

        total_coupling = attraction - h * homophily
        dX[i] = omega[i] * xi_perp + (g / k_i) * total_coupling

    # Renormalize to unit circle
    X_new = X + dt * dX
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    X_new /= np.maximum(norms, 1e-12)
    
    return X_new


# ============================================================
# 2. MACRO SIMULATION (SCALE-FREE NETWORK)
# ============================================================

def macro_simulation_2d(
        N=2000, g=1.4, h=0.92, T=3000, dt=0.01, sample_every=50,
        plot=True, seed=42):
    """
    Large-scale 2D simulation with unified diagnostics.
    """
    rng = default_rng(seed)
    G = nx.barabasi_albert_graph(N, 3)

    # Initial conditions
    theta_init = rng.uniform(0, 2*np.pi, N)
    X = np.column_stack((np.cos(theta_init), np.sin(theta_init)))
    omega = rng.normal(0, 0.2, N)

    # Storage
    R_values, VDR_inv_values = [], []
    skew_vals, kurt_vals = [], []
    chi_empirical_vals, chi_theoretical_vals = [], []
    snapshots = []

    for t in range(T):
        X = kuramoto_step_2d(X, omega, G, g, h, dt)
        
        if t % sample_every == 0:
            # Order parameter
            R_values.append(metrics.order_parameter_2d(X))
            
            # VDR^-1 (unified formula with Euclidean separation)
            VDR_inv_values.append(metrics.compute_vdr_inv(X, k_range=(2, 5), method='bic'))
            
            # Distribution moments
            sk, ku = metrics.distribution_moments(X)
            skew_vals.append(sk)
            kurt_vals.append(ku)
            
            # χ(t) - empirical and theoretical
            chi_empirical_vals.append(metrics.compute_chi_empirical(X, method='cosine'))
            
            # For theoretical χ, need F' over 2D - approximate via angles
            angles = np.arctan2(X[:, 1], X[:, 0])
            F_deriv = metrics.compute_F_derivative_mean(angles, h)
            chi_theoretical_vals.append(metrics.compute_chi_theoretical(G, g, F_deriv))
            
            snapshots.append(X.copy())

    # Compute lead time
    lt = metrics.compute_lead_time(
        np.array(VDR_inv_values), 
        np.array(R_values),
        method='percentile',
        percentile=75,
        smooth_window=5
    )

    if plot:
        fig, axs = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
        
        # (a) Order parameter
        axs[0].plot(R_values, 'b-', linewidth=2)
        axs[0].set_ylabel('R(t)', fontsize=12)
        axs[0].set_title('MACRO 2D: Order Parameter R(t)', fontsize=13, fontweight='bold')
        axs[0].grid(alpha=0.3)
        
        # (b) VDR^-1
        axs[1].plot(VDR_inv_values, 'r-', linewidth=2)
        axs[1].set_ylabel('VDR⁻¹(t)', fontsize=12)
        axs[1].set_title('MACRO 2D: VDR⁻¹(t) Early Warning', fontsize=13, fontweight='bold')
        axs[1].grid(alpha=0.3)
        
        # (c) χ(t) comparison
        axs[2].plot(chi_empirical_vals, 'g-', linewidth=2, label='χ empirical (cosine similarity)', alpha=0.8)
        axs[2].plot(chi_theoretical_vals, 'k--', linewidth=2, label='χ theoretical (true Jacobian)', alpha=0.8)
        axs[2].set_ylabel('χ(t)', fontsize=12)
        axs[2].set_title('MACRO 2D: Susceptibility χ(t) — Validation', fontsize=13, fontweight='bold')
        axs[2].legend(loc='best')
        axs[2].grid(alpha=0.3)
        
        # (d) χ correlation scatter
        valid_mask = (~np.isnan(chi_empirical_vals)) & (~np.isnan(chi_theoretical_vals))
        if np.sum(valid_mask) > 10:
            chi_emp_clean = np.array(chi_empirical_vals)[valid_mask]
            chi_theo_clean = np.array(chi_theoretical_vals)[valid_mask]
            correlation = np.corrcoef(chi_emp_clean, chi_theo_clean)[0, 1]
            
            axs[3].scatter(chi_theo_clean, chi_emp_clean, alpha=0.5, s=20)
            axs[3].plot([chi_theo_clean.min(), chi_theo_clean.max()], 
                       [chi_theo_clean.min(), chi_theo_clean.max()], 
                       'r--', linewidth=2, label='Perfect correlation')
            axs[3].set_xlabel('χ theoretical', fontsize=11)
            axs[3].set_ylabel('χ empirical', fontsize=11)
            axs[3].set_title(f'χ Correlation: ρ = {correlation:.3f}', fontsize=13, fontweight='bold')
            axs[3].legend()
            axs[3].grid(alpha=0.3)
        
        # (e) Distribution moments
        axs[4].plot(skew_vals, 'purple', linewidth=2, label='Skewness', alpha=0.8)
        axs[4].plot(kurt_vals, 'orange', linewidth=2, label='Kurtosis', alpha=0.8)
        axs[4].set_ylabel('Moments', fontsize=12)
        axs[4].set_title('Distribution Moments Over Time', fontsize=13, fontweight='bold')
        axs[4].legend(loc='best')
        axs[4].grid(alpha=0.3)
        
        # (f) Final distribution (angular)
        final_angles = np.arctan2(X[:, 1], X[:, 0])
        axs[5].hist(final_angles, bins=60, density=True, color='gray', alpha=0.8, edgecolor='black')
        axs[5].set_xlabel('θ (radians)', fontsize=11)
        axs[5].set_ylabel('Density', fontsize=11)
        axs[5].set_title('Final Angular Distribution', fontsize=13, fontweight='bold')
        axs[5].grid(alpha=0.3)
        
        axs[5].set_xlabel(f'Time step / {sample_every}', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('macro_2d_unified.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        if not np.isnan(lt):
            print(f"\n{'='*60}")
            print(f"LEAD TIME ANALYSIS (2D)")
            print(f"{'='*60}")
            print(f"VDR⁻¹ leads R(t) by: {lt:.1f} samples ({lt * sample_every:.0f} time steps)")
            print(f"{'='*60}\n")

    return {
        'X': X,
        'R': np.array(R_values),
        'VDR_inv': np.array(VDR_inv_values),
        'chi_empirical': np.array(chi_empirical_vals),
        'chi_theoretical': np.array(chi_theoretical_vals),
        'skewness': np.array(skew_vals),
        'kurtosis': np.array(kurt_vals),
        'snapshots': snapshots,
        'lead_time': lt
    }


# ============================================================
# 3. MICRO SIMULATION (Small-World Communities)
# ============================================================

def micro_simulation_2d(
        num_communities=5, N_per=150, k=6, p=0.05,
        g=0.6, h=0.32, T=1500, dt=0.02, sample_every=50,
        plot=True, seed=2, seeds_per_community=5):
    """
    Multiple 2D communities showing multistability.
    """
    rng = default_rng(seed)
    results = []

    for c in range(num_communities):
        G = nx.watts_strogatz_graph(N_per, k, p)
        community_runs = []
        
        for s in range(seeds_per_community):
            theta_init = rng.uniform(0, 2*np.pi, N_per)
            X = np.column_stack((np.cos(theta_init), np.sin(theta_init)))
            omega = rng.normal(0, 0.25, N_per)
            
            R_values, VDR_inv_values = [], []
            
            for t in range(T):
                X = kuramoto_step_2d(X, omega, G, g, h, dt)
                
                if t % sample_every == 0:
                    R_values.append(metrics.order_parameter_2d(X))
                    VDR_inv_values.append(metrics.compute_vdr_inv(X, k_range=(2, 4), method='bic'))
            
            community_runs.append((np.array(R_values), np.array(VDR_inv_values), X.copy()))
        
        results.append((G, community_runs))

    if plot:
        fig, axs = plt.subplots(num_communities, 3, figsize=(16, 4*num_communities))
        
        for c in range(num_communities):
            runs = results[c][1]
            R_mat = np.stack([r[0] for r in runs], axis=0)
            V_mat = np.stack([r[1] for r in runs], axis=0)
            
            R_mean = np.nanmean(R_mat, axis=0)
            R_std = np.nanstd(R_mat, axis=0)
            V_mean = np.nanmean(V_mat, axis=0)
            V_std = np.nanstd(V_mat, axis=0)
            
            x = np.arange(R_mean.shape[0])
            
            # R(t)
            axs[c, 0].plot(x, R_mean, color='C0', linewidth=2)
            axs[c, 0].fill_between(x, R_mean-R_std, R_mean+R_std, alpha=0.2, color='C0')
            axs[c, 0].set_ylabel('R(t)', fontsize=11)
            axs[c, 0].set_title(f'COMMUNITY {c}: R(t) mean ± std | Final R = {R_mean[-1]:.3f}', 
                               fontsize=12, fontweight='bold')
            axs[c, 0].grid(alpha=0.3)
            
            # VDR^-1
            axs[c, 1].plot(x, V_mean, color='C1', linewidth=2)
            axs[c, 1].fill_between(x, V_mean-V_std, V_mean+V_std, alpha=0.2, color='C1')
            axs[c, 1].set_ylabel('VDR⁻¹(t)', fontsize=11)
            axs[c, 1].set_title(f'COMMUNITY {c}: VDR⁻¹(t) mean ± std', fontsize=12, fontweight='bold')
            axs[c, 1].grid(alpha=0.3)
            
            # Final distribution
            final_X = runs[-1][2]
            final_angles = np.arctan2(final_X[:, 1], final_X[:, 0])
            axs[c, 2].hist(final_angles, bins=50, density=True, alpha=0.7, edgecolor='black')
            axs[c, 2].set_xlabel('θ (radians)', fontsize=11)
            axs[c, 2].set_ylabel('Density', fontsize=11)
            axs[c, 2].set_title(f'COMMUNITY {c}: Final Distribution (last seed)', 
                               fontsize=12, fontweight='bold')
            axs[c, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('micro_2d_unified.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print regime summary
        print(f"\n{'='*60}")
        print(f"MULTISTABILITY ANALYSIS (2D)")
        print(f"{'='*60}")
        for c in range(num_communities):
            runs = results[c][1]
            final_R = np.mean([r[0][-1] for r in runs])
            print(f"Community {c}: Final R = {final_R:.3f}")
        print(f"{'='*60}\n")

    return results


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("2D KURAMOTO-HOMOPHILY SIMULATION")
    print("With unified metrics (matches congressional analysis)")
    print("="*60 + "\n")
    
    print("Running macro-level 2D simulation...")
    macro_out = macro_simulation_2d(
        N=2000, g=1.4, h=0.92, T=3000, dt=0.01, 
        sample_every=50, plot=True, seed=42
    )
    
    print("\nRunning micro-level 2D communities (multistability)...")
    micro_out = micro_simulation_2d(
        num_communities=5, N_per=150, g=0.6, h=0.32,
        T=1500, dt=0.02, sample_every=50, plot=True, 
        seed=7, seeds_per_community=5
    )
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
