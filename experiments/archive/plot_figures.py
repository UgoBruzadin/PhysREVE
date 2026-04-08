"""
Presentation figures for PhysREVE EXP_003.

Two functions:
  plot_leadfield_hero()  -- brain/electrode/B_sim/channel-mixing visualization
  plot_loso_results()    -- LOSO bar charts (per-subject + mean±std)

Both return the matplotlib Figure and save a PNG if `save_path` is given.

Usage:
    python plot_figures.py                  # saves both PNGs to current dir
    python plot_figures.py --out figures/   # saves to figures/
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle


# ── Leadfield hero ───────────────────────────────────────────────────────────

def plot_leadfield_hero(save_path=None, seed=42):
    """
    Dark-mode visualization of the EEG forward model.

    Shows:
      - Head outline with brain, cortical folds
      - Four dipole sources (primary motor, left/right motor, somatosensory)
      - Ripple rings propagating from primary motor cortex
      - 22 electrodes coloured by leadfield coupling to motor cortex
      - Bezier coupling lines from motor cortex to top-8 electrodes
      - B_sim matrix heatmap (electrode-electrode cosine similarity)
      - Channel mixing bar chart for C3

    Parameters
    ----------
    save_path : str or None
        If given, saves figure as PNG at this path.
    seed : int
        Random seed for jitter in coupling lines and cortical folds.

    Returns
    -------
    fig : matplotlib.Figure
    """
    np.random.seed(seed)

    fig = plt.figure(figsize=(18, 10), facecolor='#0a0e1a')
    ax  = fig.add_axes([0, 0, 0.62, 1])
    ax.set_facecolor('#0a0e1a')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ── Head outline ─────────────────────────────────────────────────────────
    hx, hy, hr = 5.2, 4.9, 3.8
    t = np.linspace(0, 2*np.pi, 300)
    ax.fill(hx + hr*np.cos(t), hy + hr*np.sin(t), color='#1a2035', zorder=1)
    ax.plot(hx + hr*np.cos(t), hy + hr*np.sin(t), color='#4a6fa5', lw=2.5, zorder=2)
    # Nose
    ax.plot([hx-0.2, hx, hx+0.2],
            [hy+hr-0.1, hy+hr+0.35, hy+hr-0.1],
            color='#4a6fa5', lw=2.5, zorder=2)
    # Ears
    for sign in [-1, 1]:
        et = np.linspace(-0.3, 0.3, 30)
        ax.plot(hx + sign*(hr + 0.15*np.cos(et)),
                hy + np.sin(et)*0.9,
                color='#4a6fa5', lw=2.5, zorder=2)

    # ── Brain ────────────────────────────────────────────────────────────────
    br = 2.8
    ax.fill(hx + br*np.cos(t), hy + br*np.sin(t),
            color='#12203a', alpha=0.9, zorder=3)
    # Cortical folds
    for i in range(7):
        a = i * np.pi/3.5 + 0.1
        for r_ in [br*0.55, br*0.78]:
            at = np.linspace(a, a+0.75, 40)
            ax.plot(hx + r_*np.cos(at), hy + r_*np.sin(at),
                    color='#1e3a5f', lw=1.1, alpha=0.55, zorder=4)

    # ── Dipole sources ────────────────────────────────────────────────────────
    sources = [
        (5.2, 7.1, '#ff6b6b', 'Primary\nMotor (M1)', 1.0),
        (3.6, 6.5, '#ff9f43', 'Left\nMotor',         0.65),
        (6.8, 6.5, '#feca57', 'Right\nMotor',        0.55),
        (5.2, 4.3, '#48dbfb', 'Somatosensory',       0.35),
    ]
    for sx, sy, sc, lbl, _ in sources:
        for rg, ag in [(0.24, 0.07), (0.15, 0.14), (0.09, 0.32)]:
            ax.add_patch(Circle((sx, sy), rg, color=sc, alpha=ag, zorder=5))
        ax.add_patch(Circle((sx, sy), 0.065, color=sc, zorder=6))
        ax.text(sx, sy-0.30, lbl, ha='center', va='top',
                fontsize=7, color=sc, fontweight='bold', zorder=7, alpha=0.95)

    # ── Electrodes ───────────────────────────────────────────────────────────
    n_elec = 22
    e_angs = np.linspace(0, 2*np.pi, n_elec, endpoint=False) + np.pi/2
    er     = hr * 0.97
    e_pos  = [(hx + er*np.cos(a), hy + er*np.sin(a)) for a in e_angs]
    e_names = ['Fz','FC3','FC1','FCz','FC2','FC4',
               'C5','C3','C1','Cz','C2','C4','C6',
               'CP3','CP1','CPz','CP2','CP4',
               'P1','Pz','P2','POz']

    # Coupling to primary motor cortex source
    msx, msy = sources[0][0], sources[0][1]
    coupling = np.array([np.exp(-((ex-msx)**2 + (ey-msy)**2) / 3.0)
                         for ex, ey in e_pos])
    coupling = (coupling - coupling.min()) / (coupling.max() - coupling.min())
    cmap_e   = plt.cm.plasma

    for (ex, ey), name, coup in zip(e_pos, e_names, coupling):
        ec = cmap_e(coup)
        ax.add_patch(Circle((ex, ey), 0.14, color=ec, zorder=9))
        ax.add_patch(Circle((ex, ey), 0.14, color='white', fill=False, lw=0.9, zorder=10))
        if name in ['Cz','C3','C4','Fz','Pz','CPz','FCz','C1','C2']:
            ox = (ex-hx)*0.28
            oy = (ey-hy)*0.28
            ax.text(ex+ox, ey+oy, name, ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold', zorder=11,
                    path_effects=[pe.withStroke(linewidth=2, foreground='#0a0e1a')])

    # ── Ripple rings from M1 ──────────────────────────────────────────────────
    for r_ring, a_ring, ls in [(0.4,0.85,'-'), (0.7,0.6,'-'), (1.05,0.38,'-'),
                                (1.45,0.20,'--'), (1.90,0.10,'--'), (2.40,0.05,'--')]:
        rt = np.linspace(0, 2*np.pi, 250)
        ax.plot(msx + r_ring*np.cos(rt), msy + r_ring*np.sin(rt),
                color='#ff6b6b', alpha=a_ring, lw=1.3, ls=ls, zorder=8)

    # ── Coupling lines to top-8 electrodes ───────────────────────────────────
    top8 = sorted(range(n_elec), key=lambda i: -coupling[i])[:8]
    for idx in top8:
        ex, ey = e_pos[idx]
        coup_v = coupling[idx]
        mx_ = (msx+ex)/2 + np.random.uniform(-0.15, 0.15)
        my_ = (msy+ey)/2 + np.random.uniform(-0.10, 0.10)
        tc  = np.linspace(0, 1, 60)
        cx_ = (1-tc)**2*msx + 2*(1-tc)*tc*mx_ + tc**2*ex
        cy_ = (1-tc)**2*msy + 2*(1-tc)*tc*my_ + tc**2*ey
        ax.plot(cx_, cy_, color='#ff6b6b',
                alpha=coup_v*0.65, lw=1.6*coup_v, zorder=7)

    # ── Annotations ──────────────────────────────────────────────────────────
    ax.annotate('Skull\n(volume conduction)',
                xy=(hx + hr*np.cos(2.0), hy + hr*np.sin(2.0)),
                xytext=(1.0, 8.4), color='#8899aa', fontsize=8.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='#4a6fa5', lw=1.3), zorder=12)
    ax.annotate('Neural sources\n(dipole generators)',
                xy=(msx, msy+0.12), xytext=(0.8, 5.6),
                color='#ff6b6b', fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.6), zorder=12)

    # Electrode coupling colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_e, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax1 = fig.add_axes([0.01, 0.18, 0.012, 0.22])
    cb1  = fig.colorbar(sm, cax=cax1)
    cb1.ax.tick_params(labelsize=6, colors='#8899aa')
    cb1.set_label('Leadfield\ncoupling\nto M1', color='#8899aa', fontsize=7, labelpad=3)

    # ── B_sim matrix panel ────────────────────────────────────────────────────
    ax_b = fig.add_axes([0.63, 0.52, 0.17, 0.40])
    ax_b.set_facecolor('#0a0e1a')

    B = np.zeros((22, 22))
    for i in range(22):
        for j in range(22):
            xi, yi = e_pos[i]
            xj, yj = e_pos[j]
            d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            B[i, j] = (0.5 * coupling[i] * coupling[j] +
                       0.5 * np.exp(-d**2 / 4.0))
    B = (B - B.min()) / (B.max() - B.min()) * 2 - 1

    im = ax_b.imshow(B, cmap='RdBu_r', vmin=-0.5, vmax=1.0, aspect='auto')
    ax_b.set_title('B_sim = L_row @ L_rowᵀ', color='white',
                   fontsize=9, fontweight='bold', pad=5)
    ax_b.set_xlabel('Electrode index', color='#8899aa', fontsize=7.5)
    ax_b.set_ylabel('Electrode index', color='#8899aa', fontsize=7.5)
    ax_b.tick_params(colors='#8899aa', labelsize=6)
    for sp in ax_b.spines.values():
        sp.set_edgecolor('#2a3a5a')
    cb2 = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=6, colors='#8899aa')

    # ── Channel mixing bar chart ──────────────────────────────────────────────
    ax_m = fig.add_axes([0.63, 0.06, 0.34, 0.40])
    ax_m.set_facecolor('#0f172a')
    for sp in ax_m.spines.values():
        sp.set_edgecolor('#2a3a5a')

    chans = ['C3 (self)', 'C1', 'Cz', 'FCz', 'CPz', 'C4', 'O1', 'Fz']
    vals  = [1.00, 0.90, 0.72, 0.55, 0.48, -0.18, 0.04, 0.22]
    cols  = ['#ff6b6b' if v > 0 else '#4a9eff' for v in vals]

    ax_m.barh(range(len(chans)), vals, color=cols, alpha=0.87, height=0.62)
    ax_m.axvline(0, color='#4a6fa5', lw=1.3, ls='--')
    ax_m.set_yticks(range(len(chans)))
    ax_m.set_yticklabels(chans, fontsize=10, color='white', fontweight='bold')
    ax_m.set_xlabel('B_sim[C3, k] — physics coupling weight',
                    color='#8899aa', fontsize=9)
    ax_m.set_xlim(-0.45, 1.25)
    ax_m.tick_params(colors='#8899aa', labelsize=8)
    ax_m.set_title("F_out[C3]  =  F[C3]  +  α · Σₖ B_sim[C3,k] · F[k]",
                   color='white', fontsize=10, fontweight='bold', pad=8)
    for i, (_, v) in enumerate(zip(chans, vals)):
        ax_m.text(v + (0.03 if v >= 0 else -0.03), i,
                  f'{v:+.2f}', va='center',
                  ha='left' if v >= 0 else 'right',
                  fontsize=9.5, color='white', fontweight='bold')
    ax_m.text(0.5, -0.22,
              'C3 shares motor cortex sources with C1, Cz → high coupling\n'
              'O1 sees occipital sources only → near-zero coupling',
              ha='center', va='top', transform=ax_m.transAxes,
              fontsize=8.5, color='#8899aa', style='italic')

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.31, 0.975,
             'EEG Leadfield — Neural Sources to Electrode Coupling',
             ha='center', va='top', fontsize=17, color='white', fontweight='bold')
    fig.text(0.31, 0.942,
             'B_sim[i,j] = cosine similarity of electrodes i & j through the '
             'forward model — fixed by physics, not learned from data',
             ha='center', va='top', fontsize=9.5, color='#6a8fc0', style='italic')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0a0e1a', edgecolor='none')
        print(f'Saved: {save_path}')

    return fig


# ── LOSO results ─────────────────────────────────────────────────────────────

def plot_loso_results(
    A=None, B=None, C=None,
    labels=('A\nBaseline\n(REVE only)', 'B\n+Leadfield\nB_sim', 'C\nB_sim\n+Jitter'),
    colors=('#64748b', '#38bdf8', '#a78bfa'),
    save_path=None,
):
    """
    Two-panel LOSO results figure.

    Left panel:  per-subject grouped bar chart with delta annotations.
    Right panel: mean ± std summary with percentage gain arrows.

    Parameters
    ----------
    A, B, C : list of float
        Per-subject balanced accuracies for each condition.
        Defaults to EXP_003 results if not supplied.
    labels : tuple of str
        X-axis labels for the right panel (one per condition).
    colors : tuple of str
        Bar colors for conditions A, B, C.
    save_path : str or None
        If given, saves figure as PNG at this path.

    Returns
    -------
    fig : matplotlib.Figure
    """
    # EXP_003 defaults
    if A is None:
        A = [0.356, 0.323, 0.391, 0.283, 0.260, 0.359, 0.347, 0.318, 0.408]
    if B is None:
        B = [0.451, 0.311, 0.502, 0.337, 0.281, 0.314, 0.375, 0.429, 0.495]
    if C is None:
        C = [0.370, 0.321, 0.464, 0.328, 0.300, 0.363, 0.361, 0.411, 0.453]

    n_subjects = len(A)
    subjects   = list(range(1, n_subjects + 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f172a')
    fig.patch.set_facecolor('#0f172a')

    # ── Left panel: per-subject ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#0f172a')
    x = np.arange(n_subjects)
    w = 0.26

    ax.bar(x-w, A, w, label=labels[0], color=colors[0], alpha=0.85, edgecolor='#1e293b')
    ax.bar(x,   B, w, label=labels[1], color=colors[1], alpha=0.85, edgecolor='#1e293b')
    ax.bar(x+w, C, w, label=labels[2], color=colors[2], alpha=0.85, edgecolor='#1e293b')

    ax.axhline(0.25, color='#ef4444', lw=1.8, ls='--', alpha=0.8, label='Chance (0.25)')
    for cond, col in zip([A, B, C], colors):
        ax.axhline(np.mean(cond), color=col, lw=1.5, ls=':', alpha=0.7)

    # Delta annotations for B gains > 0.05
    for i, (a, b) in enumerate(zip(A, B)):
        delta = b - a
        if delta > 0.05:
            ax.annotate(f'+{delta:.2f}', xy=(x[i], b+0.005),
                        ha='center', va='bottom',
                        fontsize=8, color=colors[1], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in subjects], color='#94a3b8', fontsize=10)
    ax.set_ylabel('Balanced Accuracy', color='#94a3b8', fontsize=11)
    ax.set_title('Per-Subject Performance — LOSO',
                 color='white', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(colors='#94a3b8')
    ax.set_ylim(0.20, 0.56)
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=9,
              facecolor='#1e293b', edgecolor='#334155')
    for sp in ax.spines.values():
        sp.set_edgecolor('#334155')
    ax.yaxis.grid(True, color='#1e3a5f', alpha=0.4, lw=0.8)
    ax.set_axisbelow(True)

    # ── Right panel: mean ± std ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#0f172a')

    means = [np.mean(A), np.mean(B), np.mean(C)]
    stds  = [np.std(A),  np.std(B),  np.std(C)]

    ax2.bar(range(3), means, color=colors, alpha=0.87,
            edgecolor='#1e293b', width=0.5)
    ax2.errorbar(range(3), means, yerr=stds,
                 fmt='none', color='white', capsize=8, lw=2.5, capthick=2.5)
    ax2.axhline(0.25, color='#ef4444', lw=1.8, ls='--', alpha=0.8, label='Chance')

    # Value labels above bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax2.text(i, m + s + 0.012, f'{m:.3f}',
                 ha='center', va='bottom',
                 fontsize=14, color='white', fontweight='bold')

    # Percentage gain arrows vs baseline
    for xi, (to_mean, col, pct_label) in enumerate([
        (means[1], colors[1], f'+{(means[1]/means[0]-1)*100:.1f}%'),
        (means[2], colors[2], f'+{(means[2]/means[0]-1)*100:.1f}%'),
    ]):
        arrow_x = 1 + xi * 0.5
        ax2.annotate('', xy=(arrow_x, to_mean), xytext=(arrow_x, means[0]),
                     arrowprops=dict(arrowstyle='->', color=col, lw=2.0))
        ax2.text(arrow_x + 0.12, (means[0]+to_mean)/2,
                 pct_label, va='center',
                 fontsize=9, color=col, fontweight='bold')

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, color='white', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Balanced Accuracy (9 subjects)', color='#94a3b8', fontsize=11)
    b_wins = sum(b > a for a, b in zip(A, B))
    ax2.set_title(f'Mean ± Std — B beats A on {b_wins}/{n_subjects} subjects',
                  color='white', fontsize=13, fontweight='bold', pad=10)
    ax2.tick_params(colors='#94a3b8')
    ax2.set_ylim(0.20, 0.50)
    ax2.legend(framealpha=0.2, labelcolor='white', fontsize=9,
               facecolor='#1e293b', edgecolor='#334155')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#334155')
    ax2.yaxis.grid(True, color='#1e3a5f', alpha=0.4, lw=0.8)
    ax2.set_axisbelow(True)

    fig.suptitle(
        'Leadfield Injection into Frozen REVE — BCI IV 2a LOSO '
        f'({n_subjects} subjects, 4-class)',
        fontsize=15, color='white', fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0f172a', edgecolor='none')
        print(f'Saved: {save_path}')

    return fig


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PhysREVE presentation figures.')
    parser.add_argument('--out', default='.', help='Output directory (default: current dir)')
    args = parser.parse_args()

    out = args.out.rstrip('/')

    matplotlib.use('Agg')

    plot_leadfield_hero(save_path=f'{out}/leadfield_hero.png')
    plot_loso_results(save_path=f'{out}/loso_results.png')
