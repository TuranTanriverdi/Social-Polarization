# The Dynamics of Polarization: Four Phases of Attractor Collapse in Societies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18262713.svg)](https://doi.org/10.5281/zenodo.18262713)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Replication code and data for the manuscript:** *"The Dynamics of Polarization: Four Phases of Attractor Collapse in Societies"*

## Overview

Polarization is not simply ideological distance or partisan hostility—it is a **four-phase attractor-collapse process** driven by the spectral evolution of social-influence networks. We present a unified dynamical framework in which individual drift, homophilic interaction, and network susceptibility jointly determine ideological trajectories.

### Key Contributions

- **Three diagnostics** capture the geometry and stability of attractor basins:
  - **χ(t)**: Network susceptibility (dominant eigenvalue of influence matrix)
  - **VDR⁻¹(t)**: Variance-divergence ratio (geometric cluster separation)
  - **Center density**: Fraction of population in ideological center

- **Empirical validation** using complete U.S. Congressional roll-call history (1789–2024; N=50,829 member-terms)

- **Simulation validation** in 1D and 2D Kuramoto-homophily models demonstrating:
  - VDR⁻¹ as early-warning signal (leads R(t) by ~100 time units)
  - Stochastic multistability (identical parameters → distinct regimes)
  - Geometric-dynamical decoupling in supercritical regimes

- **Universal mechanism**: Same four-phase sequence appears across revolutions, authoritarian consolidations, and digital communities

### Four-Phase Framework

1. **Dimensional compression**: Multidimensional political space collapses onto dominant axes
2. **Geometric sorting**: Individuals migrate toward cluster centers (VDR⁻¹ peaks)
3. **Dynamical tightening**: Party discipline increases χ (even as VDR⁻¹ stabilizes)
4. **Metastable fragmentation**: Intra-party factions emerge (both χ and VDR⁻¹ decline)

---

## Repository Structure

```
.
├── README.md                              # This file
├── LICENSE                                # MIT License
│
├── Data/
│   ├── HSall_members.csv                  # Full congressional voting records (1789-2024)
│   ├── congress_diagnostics_full_final.csv    # Computed diagnostics (all 119 Congresses)
│   └── congress_diagnostics_modern_final.csv  # Computed diagnostics (90th-119th Congress)
│
├── Code/
│   ├── metrics.py                         # Unified diagnostics module (χ, VDR⁻¹, center density)
│   ├── congress_full_final.py             # Full history analysis (1789-2024, 6 eras)
│   ├── congress_modern_final.py           # Modern era analysis (1967-2024, 4 regimes)
│   ├── sim_1d.py                          # 1D Kuramoto-homophily simulations
│   └── sim_2d.py                          # 2D Kuramoto-homophily simulations
│
└── Figures/
    ├── congress_full_history_paper.png    # Six-era trajectory (1789-2024)
    ├── congress_modern_paper.png          # Four-regime trajectory (1967-2024)
    ├── macro_1d_enhanced.png              # Large-scale 1D simulation (N=2000)
    ├── macro_2d_unified.png               # Large-scale 2D simulation (N=2000)
    ├── micro_1d_enhanced.png              # Multistability demo (5 communities, 1D)
    └── micro_2d_unified.png               # Multistability demo (5 communities, 2D)
```

---

## Installation & Requirements

### Python Dependencies

```bash
pip install numpy pandas matplotlib scipy scikit-learn networkx
```

### Required packages:
- `numpy` ≥ 1.20
- `pandas` ≥ 1.3
- `matplotlib` ≥ 3.4
- `scipy` ≥ 1.7
- `scikit-learn` ≥ 1.0
- `networkx` ≥ 2.6

---

## Usage

### 1. Congressional Data Analysis

#### Full History (1789–2024): Six Eras

```bash
python congress_full_final.py
```

**Outputs:**
- `congress_full_history_paper.png`: Visualization of six eras
- `congress_diagnostics_full_final.csv`: Time series of χ(t), VDR⁻¹(t), center density

**Six identified eras:**
1. **Era 1 (1st–63rd, 1789–1915)**: High volatility, episodic realignments
2. **Era 2 (63rd–71st, 1913–1931)**: χ minimum (Republican dominance, internal heterogeneity)
3. **Era 3 (71st–85th, 1929–1957)**: χ recovery, New Deal initiates 91-year sorting
4. **Era 4 (85th–96th, 1957–1979)**: Civil Rights fragmentation
5. **Era 5 (96th–114th, 1979–2017)**: Modern polarization, geometric-dynamical decoupling
6. **Era 6 (115th–119th, 2017–present)**: Emergent intra-party factionalism

---

#### Modern Era (1967–2024): Four Regimes

```bash
python congress_modern_final.py
```

**Outputs:**
- `congress_modern_paper.png`: Visualization of four regimes
- `congress_diagnostics_modern_final.csv`: High-resolution modern diagnostics

**Four identified regimes:**
1. **Pluralistic (90th–103rd, 1967–1993)**: χ oscillates, center density elevated (mean 0.46)
2. **Sharp transition (103rd–104th, 1993–1997)**: Gingrich Revolution, center drops (0.37→0.30)
3. **Deep polarization (104th–115th, 1997–2017)**: χ rises to 0.83, VDR⁻¹ declines (decoupling)
4. **Emergent factionalism (115th–119th, 2017–present)**: Both χ and VDR⁻¹ decline simultaneously

---

### 2. Kuramoto-Homophily Simulations

#### 1D Simulations

```bash
python sim_1d.py
```

**Outputs:**
- `macro_1d_enhanced.png`: Scale-free network (N=2000), strong coupling (g=1.4, h=0.9)
  - Demonstrates sigmoid R(t) rise, VDR⁻¹ early-warning signal (~100 time-unit lead)
  - Susceptibility proxy χ(t) peaks then declines (brittle synchronization)

- `micro_1d_enhanced.png`: Five small-world communities (N=150 each, identical parameters)
  - Demonstrates stochastic multistability (final R ∈ [0.13, 0.25])

---

#### 2D Simulations

```bash
python sim_2d.py
```

**Outputs:**
- `macro_2d_unified.png`: Scale-free network (N=2000), strong coupling (g=1.4, h=0.92)
  - Shows dimensional frustration effect (slower, linear R(t) rise vs. 1D sigmoid)
  - VDR⁻¹ lead time reduced to ~40 samples (dimensional scaling)
  - χ empirical proxy validation (correlation with true Jacobian)

- `micro_2d_unified.png`: Five small-world communities (N=150 each, identical parameters)
  - Compressed multistability range (final R ∈ [0.23, 0.33] vs. 1D [0.13, 0.25])

---

## Key Findings

### Empirical (U.S. Congress)

1. **Civil War maximum**: VDR⁻¹ = 0.939 at 38th Congress (1863), highest in 235 years
2. **New Deal inflection**: 73rd Congress (1933) begins 91-year center density decline
3. **Geometric sorting peak**: 96th Congress (1979) shows VDR⁻¹ = 0.40
4. **Dynamical maximum**: 112th Congress (2010, Tea Party) shows χ = 0.83
5. **Decoupling confirmed**: VDR⁻¹ peaks 31 years before χ peaks

### Theoretical (Simulations)

1. **VDR⁻¹ is robust early-warning signal**: Leads R(t) by ~100 time units (1D), ~40 samples (2D)
2. **Multistability is universal**: Identical parameters yield distinct regimes across seeds
3. **Dimensional frustration**: 2D reduces final R by ~20% vs. 1D (geometric coordination cost)
4. **Distribution moments**: Kurtosis ≈ -1.45 is dimension-independent (universal attractor statistics)

---

## Data Sources

### Congressional Voting Records

- **Source**: Voteview DW-NOMINATE scores ([https://voteview.com/](https://voteview.com/))
- **Citation**: Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet (2023). *Voteview: Congressional Roll-Call Votes Database.* [https://voteview.com/](https://voteview.com/)
- **Coverage**: 1st–119th Congress (1789–2025), N=50,829 member-terms
- **Variables used**:
  - `nominate_dim1`: First dimension (economic/social liberalism-conservatism)
  - `nominate_dim2`: Second dimension (historical: regional/foreign policy)
  - `congress`: Congress number
  - `chamber`: House or Senate

### License

Data is used under Voteview's CC BY 4.0 license. See [https://voteview.com/about](https://voteview.com/about) for details.

---

## Replication Instructions

### Full Replication (all analyses)

```bash
# Congressional analysis
python congress_full_final.py      # ~5 minutes
python congress_modern_final.py    # ~1 minute

# Simulations
python sim_1d.py                   # ~10 minutes
python sim_2d.py                   # ~15 minutes
```

### Compute custom diagnostics

```python
import pandas as pd
import numpy as np
import metrics

# Load data
df = pd.read_csv('HSall_members.csv')
cong_df = df[df['congress'] == 112]  # Example: 112th Congress
X = cong_df[['nominate_dim1', 'nominate_dim2']].values

# Compute diagnostics
chi = metrics.compute_chi_empirical(X, method='cosine')
vdr_inv = metrics.compute_vdr_inv(X, k_range=(2, 5), method='bic')
center_dens = metrics.compute_center_density(X, center_width=0.3, dim=0)

print(f"χ(t) = {chi:.2f}")
print(f"VDR⁻¹ = {vdr_inv:.3f}")
print(f"Center density = {center_dens:.3f}")
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{authorYEAR,
  title={The Dynamics of Polarization: Four Phases of Attractor Collapse in Societies},
  author={[Author Name]},
  journal={[Journal/Preprint]},
  year={2025},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://github.com/[username]/Social-Polarization}
}
```

**Preprint**: [Link to Zenodo] (DOI: 10.5281/zenodo.XXXXXXX)

---

## Theoretical Background

### Mathematical Framework

Each individual *i* is represented by a *d*-dimensional state vector **x**_*i*(*t*) ∈ ℝ^*d* encoding political attitudes. The dynamics are:

**ẋ**_*i* = **b**_*i* + ∑_*j* *g*_*ij* **F**(**x**_*j* − **x**_*i*) + **η**_*i*(*t*)

where:
- **b**_*i*: Intrinsic drift (personality, exogenous shocks)
- *G* = (*g*_*ij*): Weighted influence matrix
- **F**: Interaction kernel (dissipative, odd, Lipschitz)
- **η**_*i*: White noise

### Critical Threshold

The system undergoes a phase transition when network susceptibility exceeds a critical value:

χ ≡ max{Re(λ) : λ ∈ σ(*J*)} ≥ χ_c ≈ 1

where *J* is the Jacobian of the interaction term. For χ > χ_c, social forces overwhelm intrinsic drift, rendering autonomous moderation dynamically unstable.

### Three Diagnostics

1. **χ(*t*)**: Empirical proxy via dominant eigenvalue of cosine similarity matrix
2. **VDR⁻¹(*t*)**: Within-cluster variance / between-cluster separation (via GMM with BIC)
3. **Center density**: Fraction within ±0.3 on first ideological dimension

---

## Contact & Contributing

- **Issues**: Please report bugs or request features via [GitHub Issues](https://github.com/[username]/Social-Polarization/issues)
- **Pull requests**: Contributions welcome (code improvements, documentation, additional analyses)
- **Contact**: turantanriverdi84@gmail.com

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data attribution**: Congressional voting data from Voteview (CC BY 4.0). See [Data Sources](#data-sources) above.

---

## Acknowledgments

- Voteview team (Lewis, Poole, Rosenthal et al.) for maintaining the DW-NOMINATE database
- [Any funding sources or institutional support]
- [Any collaborators or advisors]

---

## Changelog

### v1.0.0 (2025-01-XX)
- Initial release accompanying manuscript submission
- Full congressional analysis (1789-2024)
- 1D and 2D Kuramoto-homophily simulations
- Unified metrics module with three core diagnostics

---

**Repository**: https://github.com/[username]/Social-Polarization

**Preprint**: [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.18262713)

**Last updated**: January 2025
