"""
Unified metrics module for dynamical-attractor framework.
Used by both simulations and empirical congressional analysis.
Ensures methodological consistency across synthetic and real data.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis
from itertools import combinations
import networkx as nx


# ============================================================
# 1. CORE ORDER PARAMETERS
# ============================================================

def order_parameter_1d(theta):
    """
    Global coordination R = |mean of unit complex| in [0,1].
    For 1D angular data.
    """
    return np.abs(np.mean(np.exp(1j * theta)))


def order_parameter_2d(X):
    """
    Global coordination R = || mean vector || in [0,1].
    For 2D unit vectors.
    """
    return np.linalg.norm(np.mean(X, axis=0))


# ============================================================
# 2. VDR^-1 (UNIFIED FORMULA)
# ============================================================

def compute_vdr_inv(X, k_range=(2, 5), method='bic', random_state=42):
    """
    Unified VDR^-1 computation for both simulation and empirical data.
    
    Parameters:
    -----------
    X : array-like, shape (N, d)
        Position data (d=2 for angles projected to unit circle or congressional coords)
    k_range : tuple
        Range of cluster numbers to test (min_k, max_k)
    method : str
        'bic' or 'silhouette' for cluster selection
    
    Returns:
    --------
    vdr_inv : float
        VDR^-1 = within-cluster variance / mean pairwise separation
    """
    if len(X) < 20:
        return np.nan
    
    best_gmm = None
    best_score = np.inf if method == 'bic' else -np.inf
    best_k = k_range[0]
    
    for k in range(k_range[0], min(k_range[1], len(X)//10 + 1)):
        gmm = GaussianMixture(n_components=k, covariance_type='full', 
                             random_state=random_state, max_iter=100)
        try:
            gmm.fit(X)
            
            if method == 'bic':
                score = gmm.bic(X)
                if score < best_score:
                    best_score = score
                    best_gmm = gmm
                    best_k = k
            elif method == 'silhouette' and k > 1:
                labels = gmm.predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_gmm = gmm
                    best_k = k
        except:
            continue
    
    if best_gmm is None or best_k < 2:
        return np.nan
    
    means = best_gmm.means_
    covs = best_gmm.covariances_
    
    # Within-cluster variance: mean trace of covariances
    within = np.mean([np.trace(cov) for cov in covs])
    
    # Between-cluster separation: mean pairwise squared distance
    k = len(means)
    pairwise_dists = [np.linalg.norm(means[i] - means[j])**2 
                      for i, j in combinations(range(k), 2)]
    
    if len(pairwise_dists) == 0:
        return np.nan
    
    between = np.mean(pairwise_dists)
    
    return within / between if between > 1e-6 else np.nan


# ============================================================
# 3. NETWORK SUSCEPTIBILITY χ(t)
# ============================================================

def compute_chi_empirical(X, method='cosine', bandwidth=None):
    """
    Empirical susceptibility from position data (when true network unknown).
    
    Parameters:
    -----------
    X : array-like, shape (N, d)
        Position data
    method : str
        'cosine' or 'rbf' similarity kernel
    bandwidth : float or None
        For RBF kernel; if None, uses Scott's rule
    
    Returns:
    --------
    chi : float
        Dominant eigenvalue of similarity matrix (synchronization potential)
    """
    if method == 'cosine':
        G = cosine_similarity(X)
    elif method == 'rbf':
        if bandwidth is None:
            # Scott's rule
            bandwidth = len(X) ** (-1.0 / (X.shape[1] + 4))
        G = rbf_kernel(X, gamma=1.0/(2*bandwidth**2))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    eigvals = np.linalg.eigvalsh(G)
    return np.max(eigvals)


def compute_chi_theoretical(G_network, g, F_derivative_mean):
    """
    Theoretical susceptibility when true network is known (simulation only).
    
    Parameters:
    -----------
    G_network : networkx.Graph or numpy.ndarray
        True interaction network
    g : float
        Global coupling strength
    F_derivative_mean : float
        Mean value of |F'(state)| over current configuration
    
    Returns:
    --------
    chi : float
        Spectral radius of effective Jacobian: g * λ_max(A) * <|F'|>
    """
    if isinstance(G_network, nx.Graph):
        A = nx.to_numpy_array(G_network)
    else:
        A = G_network
    
    lam_max = np.linalg.eigvalsh(A).max()
    return g * lam_max * F_derivative_mean


def compute_chi_laplacian(X, threshold=0.1):
    """
    Alternative empirical susceptibility via algebraic connectivity.
    χ ≈ 1 / (Fiedler value) measures fragmentation tendency.
    
    Parameters:
    -----------
    X : array-like, shape (N, d)
        Position data
    threshold : float
        Similarity threshold for sparsification
    
    Returns:
    --------
    chi : float
        Inverse of second-smallest Laplacian eigenvalue
    """
    G = cosine_similarity(X)
    G[G < threshold] = 0  # sparsify
    
    # Row-normalize
    row_sums = G.sum(axis=1, keepdims=True)
    G_norm = G / (row_sums + 1e-12)
    
    # Laplacian
    D = np.diag(G_norm.sum(axis=1))
    L = D - G_norm
    
    eigvals = np.linalg.eigvalsh(L)
    eigvals_sorted = np.sort(np.real(eigvals))
    
    # Fiedler value (algebraic connectivity)
    fiedler = eigvals_sorted[1] if len(eigvals_sorted) > 1 else eigvals_sorted[0]
    
    return 1.0 / fiedler if fiedler > 1e-6 else np.inf


# ============================================================
# 4. CENTER DENSITY (EARLY WARNING)
# ============================================================

def compute_center_density(X, center_width=0.3, dim=0, adaptive=False):
    """
    Density of ideological center (early warning of polarization).
    
    Parameters:
    -----------
    X : array-like, shape (N, d)
        Position data
    center_width : float
        Half-width of center region (if not adaptive)
    dim : int
        Dimension to use for center definition (typically 0 for left-right)
    adaptive : bool
        If True, define center as middle 20% of observed range
    
    Returns:
    --------
    density : float
        Fraction of population in center region
    """
    x_dim = X[:, dim] if X.ndim > 1 else X
    
    if adaptive:
        x_range = x_dim.max() - x_dim.min()
        center_width_adapt = 0.1 * x_range  # center = middle 20% of range
        x_median = np.median(x_dim)
        in_center = np.abs(x_dim - x_median) < center_width_adapt
    else:
        in_center = np.abs(x_dim) < center_width
    
    return np.sum(in_center) / len(x_dim)


# ============================================================
# 5. DISTRIBUTION MOMENTS
# ============================================================

def distribution_moments(X):
    """
    Skewness and kurtosis for attractor shape characterization.
    For circular data (1D angles), projects to Cartesian to avoid wrap artifacts.
    
    Parameters:
    -----------
    X : array-like, shape (N,) or (N, d)
        If 1D angles, computes over cos/sin projections
        If multidimensional, computes over all dimensions
    
    Returns:
    --------
    skewness, kurtosis : float
        Averaged over dimensions
    """
    if X.ndim == 1:
        # Assume circular data - project to avoid discontinuities
        x_proj = np.cos(X)
        y_proj = np.sin(X)
        sk = 0.5 * (skew(x_proj) + skew(y_proj))
        ku = 0.5 * (kurtosis(x_proj, fisher=True) + kurtosis(y_proj, fisher=True))
    else:
        # Multidimensional data
        sk = np.mean([skew(X[:, i]) for i in range(X.shape[1])])
        ku = np.mean([kurtosis(X[:, i], fisher=True) for i in range(X.shape[1])])
    
    return sk, ku


# ============================================================
# 6. ONSET DETECTION AND LEAD TIME
# ============================================================

def detect_onset(series, method='percentile', percentile=75, smooth_window=5):
    """
    Detect onset of rapid increase in time series.
    
    Parameters:
    -----------
    series : array-like
        Time series to analyze
    method : str
        'percentile' (cross threshold) or 'derivative' (sustained increase)
    percentile : float
        For 'percentile' method
    smooth_window : int
        Window for moving average smoothing
    
    Returns:
    --------
    onset_idx : int or nan
        Index of onset, or nan if not detected
    """
    series = np.array(series, dtype=float)
    
    if len(series) < smooth_window + 2:
        return np.nan
    
    # Remove nans
    valid_mask = ~np.isnan(series)
    if np.sum(valid_mask) < smooth_window:
        return np.nan
    
    series_clean = series[valid_mask]
    
    # Smooth
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        series_smooth = np.convolve(series_clean, kernel, mode='valid')
    else:
        series_smooth = series_clean
    
    if method == 'percentile':
        threshold = np.percentile(series_smooth, percentile)
        crossings = np.where(series_smooth > threshold)[0]
        return crossings[0] if len(crossings) > 0 else np.nan
    
    elif method == 'derivative':
        ds = np.diff(series_smooth)
        threshold = np.mean(ds) + np.std(ds)
        crossings = np.where(ds > threshold)[0]
        return crossings[0] + (smooth_window - 1) if len(crossings) > 0 else np.nan
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_lead_time(early_signal, late_signal, method='percentile', **kwargs):
    """
    Compute lead time between two signals (e.g., VDR^-1 and R).
    
    Parameters:
    -----------
    early_signal : array-like
        Expected leading indicator (e.g., VDR^-1)
    late_signal : array-like
        Expected lagging indicator (e.g., R)
    method : str
        Onset detection method
    **kwargs : dict
        Additional arguments for detect_onset
    
    Returns:
    --------
    lead_time : float or nan
        Time steps between onsets (negative = early_signal lags)
    """
    early_onset = detect_onset(early_signal, method=method, **kwargs)
    late_onset = detect_onset(late_signal, method=method, **kwargs)
    
    if np.isnan(early_onset) or np.isnan(late_onset):
        return np.nan
    
    return late_onset - early_onset


# ============================================================
# 7. F DERIVATIVE FOR KURAMOTO-HOMOPHILY MODEL
# ============================================================

def compute_F_derivative_mean(theta, h):
    """
    Mean absolute derivative of F(Δθ) = sin(Δθ) - h·sin(2Δθ)
    F'(Δθ) = cos(Δθ) - 2h·cos(2Δθ)
    
    For theoretical χ in Kuramoto-homophily simulations.
    
    Parameters:
    -----------
    theta : array-like
        Current angular positions
    h : float
        Homophily parameter
    
    Returns:
    --------
    F_deriv_mean : float
        Mean of |F'(Δθ)| approximated over distribution
    """
    # Approximate via mean over all angles
    return np.mean(np.abs(np.cos(theta)) + 2*h*np.abs(np.cos(2*theta)))
