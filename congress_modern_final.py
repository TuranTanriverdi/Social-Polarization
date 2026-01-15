"""
Modern U.S. Congress analysis (1967-2024) with unified diagnostics.
Final version: Raw VDR values, honest χ labeling, visual trend focus.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metrics


# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

print("Loading HSall_members.csv...")
df = pd.read_csv('HSall_members.csv')
df = df[df['congress'] >= 90]  # Start from 90th Congress (1967)
df = df.dropna(subset=['nominate_dim1', 'nominate_dim2'])

print(f"Analyzing Congresses {df['congress'].min()} → {df['congress'].max()}")
print(f"Total records: {len(df)}")


# ============================================================
# 2. COMPUTE DIAGNOSTICS FOR EACH CONGRESS
# ============================================================

chi_series = []
vdr_series = []
center_density = []
congresses = []

for cong in sorted(df['congress'].unique()):
    cong_df = df[df['congress'] == cong]
    X = cong_df[['nominate_dim1', 'nominate_dim2']].values
    
    if len(X) < 100:
        continue
    
    congresses.append(cong)
    
    # 1. χ(t) - empirical proxy (relative indicator only)
    chi_emp = metrics.compute_chi_empirical(X, method='cosine')
    chi_series.append(chi_emp)
    
    # 2. VDR^-1 with unified formula (RAW, no scaling)
    vdr_inv = metrics.compute_vdr_inv(X, k_range=(2, 5), method='bic')
    vdr_series.append(vdr_inv)
    
    # 3. Center density (early warning signal)
    center_dens = metrics.compute_center_density(X, center_width=0.3, dim=0, adaptive=False)
    center_density.append(center_dens)

print(f"\nProcessed {len(congresses)} Congresses")


# ============================================================
# 3. PAPER-READY VISUALIZATION: FOUR REGIMES (1967–2024)
# ============================================================

fig, ax1 = plt.subplots(figsize=(16, 9))

# Left y-axis: χ(t)
color_chi = 'tab:blue'
ax1.plot(congresses, chi_series, color=color_chi, linewidth=3, 
         label='χ(t) — network susceptibility', marker='o', markersize=3.5, alpha=0.9)
ax1.set_xlabel('Congress (90th = 1967–1969, 119th = 2023–2025)', fontsize=14, fontweight='bold')
ax1.set_ylabel('χ(t) proxy', fontsize=14, color=color_chi, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_chi, labelsize=12)
ax1.grid(alpha=0.25, linewidth=0.8, axis='x')

# Right y-axis: Center density + VDR⁻¹ (same scale)
ax2 = ax1.twinx()
ax2.plot(congresses, center_density, 'r-', linewidth=3.5, 
         label='Center density', marker='o', markersize=4, alpha=0.95)
ax2.plot(congresses, vdr_series, 'purple', linewidth=3, 
         label='VDR⁻¹', marker='s', markersize=3.5, alpha=0.85, linestyle='--')
ax2.set_ylabel('Center Density | VDR⁻¹', fontsize=14, color='red', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='red', labelsize=12)

# Four regimes with distinct shading (as per paper Section 10.4.1)
regime_regions = [
    (90, 103, '#D4F1D4', 0.35, 'Pluralistic\n(1967–1993)'),
    (103, 104, '#FFE4E1', 0.45, 'Sharp\ntransition'),
    (104, 115, '#FFD7B5', 0.35, 'Deep polarization\n(1997–2017)'),
    (115, max(congresses), '#E8D5F2', 0.35, 'Emergent factionalism\n(2017–)')
]

for start, end, color, alpha, label in regime_regions:
    ax1.axvspan(start, end, alpha=alpha, color=color, zorder=0)

# Regime boundary lines
regime_boundaries = [
    (103, 'green', '--', 2.5),
    (104, 'darkred', '-', 4.0),
    (115, 'purple', '-', 3.5)
]

for cong, color, style, lw in regime_boundaries:
    ax1.axvline(cong, color=color, linestyle=style, linewidth=lw, alpha=0.75, zorder=2)

# Key annotations - will position after axes limits are set
# Store annotation data for later
annotation_data = [
    # (congress, text, axis, color, relative_y_position)
    (96, 'χ oscillates\nno sustained trend', 'left', 'blue', 0.80),
    (104, 'Center: 0.37→0.30', 'right', 'darkred', 0.85),
    (108, 'Gingrich:\nχ rises', 'left', 'darkred', 0.65),
    (112, 'χ peak: 0.83', 'left', 'orange', 0.92),
    (98, 'VDR⁻¹ peak', 'right', 'purple', 0.70),
    (116, 'Both decline', 'left', 'purple', 0.75)
]

# Regime labels at top
regime_labels = [
    (96.5, 'PLURALISTIC', 'green'),
    (103.5, 'TRANS', 'darkred'),
    (109.5, 'DEEP POLARIZATION', 'darkorange'),
    (117, 'FACTIONALISM', 'purple')
]

# Add regime labels first
for cong, text, color in regime_labels:
    y_pos = ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.97
    ax1.text(cong, y_pos, text, fontsize=11, fontweight='bold', 
            color=color, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', 
                     edgecolor=color, linewidth=2.5, alpha=0.95))

# Now add annotations with proper relative positioning
for cong, text, axis, color, rel_y in annotation_data:
    ax_target = ax1 if axis == 'left' else ax2
    y_limits = ax_target.get_ylim()
    y_text = y_limits[0] + (y_limits[1] - y_limits[0]) * rel_y
    
    if cong in congresses:
        idx = congresses.index(cong)
        if axis == 'left':
            y_val = chi_series[idx] if not np.isnan(chi_series[idx]) else y_text
        else:
            # For right axis annotations
            if 'VDR' in text:
                y_val = vdr_series[idx] if not np.isnan(vdr_series[idx]) else y_text
            else:
                y_val = center_density[idx] if not np.isnan(center_density[idx]) else y_text
    else:
        y_val = y_text
    
    # Position text box near the point
    x_offset = 2 if cong < 110 else -2
    
    ax_target.annotate(text, xy=(cong, y_val), 
                      xytext=(cong + x_offset, y_text),
                      fontsize=9, fontweight='bold', color=color,
                      arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.7),
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               alpha=0.92, edgecolor=color, linewidth=1.8),
                      ha='center')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
          fontsize=12, framealpha=0.95, edgecolor='black', fancybox=False)

# Title matching paper terminology
plt.title('U.S. Congress 1967–2024: Modern Era — Four Regime Transitions\n' +
         'Pluralistic (center=0.46) → Sharp transition (0.37→0.30) → Deep polarization (χ: 0.60→0.83, geometric-dynamical decoupling) → Emergent factionalism',
         fontsize=13.5, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('congress_modern_paper.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\nSaved: congress_modern_paper.png")


# ============================================================
# 4. PRINT KEY OBSERVATIONS
# ============================================================

print("\n" + "="*60)
print("KEY OBSERVATIONS FOR VISUAL INTERPRETATION")
print("="*60)

# Find major changes
chi_diff = np.diff(chi_series)
center_diff = np.diff(center_density)
vdr_diff = np.diff(vdr_series)

# Largest χ increases
chi_rises = [(congresses[i], chi_diff[i]) for i in np.argsort(chi_diff)[-3:]]
print("\nLargest χ(t) increases:")
for cong, diff in sorted(chi_rises, reverse=True):
    print(f"  Congress {cong}→{cong+1}: +{diff:.1f}")

# Largest center density drops
center_drops = [(congresses[i], center_diff[i]) for i in np.argsort(center_diff)[:3]]
print("\nLargest center density drops:")
for cong, diff in sorted(center_drops, key=lambda x: x[1]):
    print(f"  Congress {cong}→{cong+1}: {diff:.3f}")

# Largest VDR spikes
vdr_clean = [v if not np.isnan(v) else 0 for v in vdr_series]
vdr_peaks = [(congresses[i], vdr_clean[i]) for i in np.argsort(vdr_clean)[-5:]]
print("\nHighest VDR⁻¹ values:")
for cong, val in sorted(vdr_peaks, key=lambda x: x[1], reverse=True):
    print(f"  Congress {cong}: VDR⁻¹ = {val:.3f}")


# ============================================================
# 5. SAVE PROCESSED DATA
# ============================================================

results_df = pd.DataFrame({
    'congress': congresses,
    'chi_proxy': chi_series,
    'vdr_inv_raw': vdr_series,
    'center_density': center_density
})

results_df.to_csv('congress_diagnostics_modern_final.csv', index=False)
print("Saved diagnostics to: congress_diagnostics_modern_final.csv\n")
