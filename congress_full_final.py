"""
Full U.S. Congress history (1789-2024) with unified diagnostics.
Final version: Raw VDR, honest χ, focus on visual trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import metrics


# ============================================================
# 1. LOAD FULL DATASET
# ============================================================

print("Loading HSall_members.csv (full history 1789-2024)...")
df = pd.read_csv('HSall_members.csv')
df = df.dropna(subset=['nominate_dim1', 'nominate_dim2'])

print(f"Total valid records: {len(df)}")
print(f"Congress range: {df['congress'].min()} → {df['congress'].max()}")


# ============================================================
# 2. COMPUTE DIAGNOSTICS FOR ALL CONGRESSES
# ============================================================

congresses = sorted(df['congress'].unique())
print(f"Found {len(congresses)} Congresses with data")

chi_series = []
center_density = []
vdr_series = []
num_members = []

for cong in congresses:
    cong_df = df[df['congress'] == cong]
    X = cong_df[['nominate_dim1', 'nominate_dim2']].values
    
    num_members.append(len(X))
    
    # Skip very small early Congresses
    if len(X) < 50:
        chi_series.append(np.nan)
        center_density.append(np.nan)
        vdr_series.append(np.nan)
        continue
    
    # 1. χ(t) - empirical proxy (normalized by size)
    chi_emp = metrics.compute_chi_empirical(X, method='cosine')
    chi_normalized = chi_emp / len(X)  # Normalize by network size
    chi_series.append(chi_normalized)
    
    # 2. Center density (adaptive for historical variation)
    center_dens = metrics.compute_center_density(X, center_width=0.3, dim=0, adaptive=True)
    center_density.append(center_dens)
    
    # 3. VDR^-1 (RAW, no scaling)
    if len(X) >= 100:
        vdr_inv = metrics.compute_vdr_inv(X, k_range=(2, 5), method='bic')
        vdr_series.append(vdr_inv)
    else:
        vdr_series.append(np.nan)

print(f"\nProcessed {len(congresses)} Congresses")


# ============================================================
# 3. PAPER-READY VISUALIZATION: SIX ERAS (1789–2024)
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

# Top panel: χ(t) network susceptibility
ax1.plot(congresses, chi_series, 'b-', linewidth=2.8, label='χ(t) — network susceptibility')
ax1.set_ylabel('χ(t) / N', fontsize=14, color='b', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b', labelsize=11)
ax1.grid(alpha=0.25, linewidth=0.8)
ax1.set_ylim([0.50, 0.82])

# Bottom panel: Center density + VDR⁻¹ (raw, same scale)
ax2.plot(congresses, center_density, 'r-', linewidth=3.5, 
         label='Center density', marker='o', markersize=2.5, alpha=0.95)
ax2.plot(congresses, vdr_series, 'purple', alpha=0.85, linewidth=2.8, 
         label='VDR⁻¹', marker='s', markersize=2.5, linestyle='--')
ax2.set_ylabel('Density / VDR⁻¹', fontsize=14, fontweight='bold')
ax2.set_xlabel('Congress (1st = 1789–1791, 119th = 2023–2025)', fontsize=14, fontweight='bold')
ax2.tick_params(labelsize=11)
ax2.grid(alpha=0.25, linewidth=0.8)
ax2.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='black', fancybox=False)

# Six eras with distinct shading (as per paper Section 10.4.2)
era_regions = [
    (1, 63, '#F5E6D3', 0.35, 'Era 1 (1789–1915)\nHigh volatility'),
    (63, 71, '#D4E4F7', 0.35, 'Era 2 (1913–1931)\nχ minimum'),
    (71, 85, '#E8F5E9', 0.35, 'Era 3 (1929–1957)\nχ recovery'),
    (85, 96, '#FFF9C4', 0.35, 'Era 4 (1957–1979)\nFragmentation'),
    (96, 115, '#FFCCBC', 0.35, 'Era 5 (1979–2017)\nModern polarization'),
    (115, max(congresses), '#E1BEE7', 0.35, 'Era 6 (2017–)\nIntra-party factions')
]

for start, end, color, alpha, label in era_regions:
    ax1.axvspan(start, end, alpha=alpha, color=color, zorder=0)
    ax2.axvspan(start, end, alpha=alpha, color=color, zorder=0)

# Era boundary lines (major transitions)
era_boundaries = [
    (63, 'Era 1|2', 'navy', ':', 2.5),
    (71, 'Era 2|3', 'darkgreen', ':', 2.5),
    (85, 'Era 3|4', 'goldenrod', ':', 2.5),
    (96, 'Era 4|5', 'orangered', '-', 3.0),
    (115, 'Era 5|6', 'purple', '-', 3.0)
]

for cong, label, color, style, lw in era_boundaries:
    for ax in [ax1, ax2]:
        ax.axvline(cong, color=color, linestyle=style, linewidth=lw, alpha=0.7, zorder=2)

# Key historical events
historical_events = [
    (38, 'Civil War\n(1863)', 'brown', 'center', (38, 0.55)),
    (67, 'χ min\n(1920s)', 'blue', 'top', (67, 0.78)),
    (73, 'New Deal\n(1933)', 'green', 'top', (73, 0.74)),
    (96, 'VDR⁻¹ peak\n(1979)', 'orange', 'bottom', (96, 0.45)),
    (112, 'χ max\n(2010)', 'red', 'top', (112, 0.78))
]

for cong, text, color, panel, xy_text in historical_events:
    if panel == 'top':
        ax = ax1
        idx = congresses.index(cong) if cong in congresses else None
        if idx is not None:
            y_val = chi_series[idx] if not np.isnan(chi_series[idx]) else 0.7
        else:
            y_val = 0.7
    elif panel == 'bottom':
        ax = ax2
        idx = congresses.index(cong) if cong in congresses else None
        if idx is not None:
            y_val = vdr_series[idx] if not np.isnan(vdr_series[idx]) else 0.3
        else:
            y_val = 0.3
    else:  # center = both
        # VDR spike on bottom panel
        ax = ax2
        idx = congresses.index(cong) if cong in congresses else None
        if idx is not None:
            y_val = vdr_series[idx] if not np.isnan(vdr_series[idx]) else 0.5
        else:
            y_val = 0.5
    
    ax.annotate(text, xy=(cong, y_val), xytext=xy_text,
                fontsize=10, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.75),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.9, edgecolor=color, linewidth=2),
                ha='center')

# Era labels at top
era_label_positions = [
    (32, 0.81, 'ERA 1', '#8B4513'),
    (67, 0.81, 'ERA 2', '#1976D2'),
    (78, 0.81, 'ERA 3', '#388E3C'),
    (90, 0.81, 'ERA 4', '#F57C00'),
    (105, 0.81, 'ERA 5', '#D32F2F'),
    (117, 0.81, 'ERA 6', '#7B1FA2')
]

for cong, y_pos, text, color in era_label_positions:
    ax1.text(cong, y_pos, text, fontsize=11, fontweight='bold', 
            color=color, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=color, linewidth=2, alpha=0.95))

# Legend for χ(t)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=False)

# Title with six-era framework
fig.suptitle('U.S. Congress 1789–2024: Six Eras of Structural Transformation\n' +
             'Era 1: High volatility | Era 2: χ minimum | Era 3: χ recovery | Era 4: Fragmentation | Era 5: Modern polarization | Era 6: Intra-party factions',
             fontsize=13, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig('congress_full_history_paper.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\nSaved: congress_full_history_paper.png")


# ============================================================
# 4. IDENTIFY KEY INFLECTION POINTS
# ============================================================

print("\n" + "="*60)
print("KEY INFLECTION POINTS FOR VISUAL INTERPRETATION")
print("="*60)

# Find major VDR spikes
vdr_clean = np.array([v if not np.isnan(v) else 0 for v in vdr_series])
vdr_top = np.argsort(vdr_clean)[-10:]
print("\nTop 10 VDR⁻¹ values (structural tightening episodes):")
for idx in vdr_top[::-1]:
    if vdr_clean[idx] > 0:
        print(f"  Congress {congresses[idx]} ({1789 + 2*congresses[idx] - 2}): VDR⁻¹ = {vdr_clean[idx]:.3f}")

# Find χ trends
chi_clean = np.array([c if not np.isnan(c) else 0 for c in chi_series])
chi_diff = np.diff(chi_clean)
print("\nχ(t) trend analysis:")
print(f"  Major rise period: ~62nd-85th Congress (χ: {chi_clean[61]:.3f} → {chi_clean[84]:.3f})")
print(f"  Decline period: ~85th-95th Congress (χ: {chi_clean[84]:.3f} → {chi_clean[94]:.3f})")
print(f"  Modern rise: ~95th-112th Congress (χ: {chi_clean[94]:.3f} → {chi_clean[111]:.3f})")
print(f"  Recent decline: ~112th-119th Congress (χ: {chi_clean[111]:.3f} → {chi_clean[-1]:.3f})")

# Center density collapse
center_clean = np.array([c if not np.isnan(c) else 0 for c in center_density])
print(f"\nCenter density evolution:")
print(f"  Early range (1st-60th): {np.nanmean(center_clean[:60]):.3f} ± {np.nanstd(center_clean[:60]):.3f}")
print(f"  Mid-century (60th-90th): {np.nanmean(center_clean[60:90]):.3f} ± {np.nanstd(center_clean[60:90]):.3f}")
print(f"  Modern (90th-119th): {np.nanmean(center_clean[90:]):.3f} ± {np.nanstd(center_clean[90:]):.3f}")

# ============================================================
# 5. SAVE FULL DIAGNOSTICS
# ============================================================

results_df = pd.DataFrame({
    'congress': congresses,
    'year_start': [1789 + 2*(c-1) for c in congresses],
    'chi_normalized': chi_series,
    'center_density': center_density,
    'vdr_inv_raw': vdr_series,
    'num_members': num_members
})

results_df.to_csv('congress_diagnostics_full_final.csv', index=False)
print("Saved full diagnostics to: congress_diagnostics_full_final.csv\n")

print("="*60)
print("ANALYSIS COMPLETE")
print("="*60)
