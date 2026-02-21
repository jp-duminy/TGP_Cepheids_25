"""
Publication-quality plotting functions for Andromeda CV1 distance inference.
Style matched to the P-L relation plots.
@author: jp

This is entirely made by Claude, alas I do not have time to fancify plots myself.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import gaussian_kde
import scienceplots

plt.style.use('science')
plt.rcParams['text.usetex'] = False


def plot_distance_posterior(distance_kpc, literature_kpc=761, savefig=False):
    """
    Publication-quality posterior distance histogram with KDE overlay.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    median = np.median(distance_kpc)
    lower = np.percentile(distance_kpc, 16)
    lower2 = np.percentile(distance_kpc, 2.5)
    upper = np.percentile(distance_kpc, 84)
    upper2=np.percentile(distance_kpc, 97.5)

    # histogram
    n, bins, patches = ax.hist(distance_kpc, bins=60, density=True,
                                color='#4878A8', edgecolor='white',
                                linewidth=0.4, alpha=0.75, zorder=2)

    # KDE overlay
    kde = gaussian_kde(distance_kpc)
    x_kde = np.linspace(np.percentile(distance_kpc, 0.5),
                        np.percentile(distance_kpc, 99.5), 300)
    ax.plot(x_kde, kde(x_kde), color="#000263", linewidth=2, zorder=3)

    # median and 1-sigma
    ax.axvline(median, color="#C44E52", linestyle='-', linewidth=2.0,
               label=rf'Median: {median:.0f}$^{{+{upper - median:.0f}}}_{{-{median - lower:.0f}}}$ kpc',
               zorder=4)
    ax.axvspan(lower, upper, color='#C44E52', alpha=0.12, zorder=1,
               label=f"68% credible interval")
    ax.axvspan(lower2, upper2, color='#C44E52', alpha=0.08, zorder=1,
               label=f"95% credible interval")

    # literature
    ax.axvline(literature_kpc, color="#0DFD79", linestyle='--', linewidth=2.0,
               label=f'Literature: 761 kPc (Siyang et al., 2021]', zorder=4)

    ax.set_xlabel('Distance [kpc]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Posterior Distance to M31 via Cepheid CV1', fontsize=13)
    ax.legend(fontsize=9, frameon=True, facecolor='white',
              edgecolor='black', framealpha=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

    if savefig:
        fig.savefig('andromeda_distance_posterior.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_on_pl_relation(period_samples, M_samples, pl_chain,
                        savefig=False):
    """
    Publication-quality P-L relation with Andromeda CV1 overlaid.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- P-L fit from chain ---
    med_period = np.median(period_samples)
    all_periods = [med_period]
    period_range = np.linspace(2, 80, 200)
    log_p = np.log10(period_range)

    draw = np.random.randint(0, len(pl_chain), size=200)
    models = np.array([pl_chain[d, 0] * (log_p - 1) + pl_chain[d, 1]
                        for d in draw])
    med_model = np.median(models, axis=0)
    spread = np.std(models, axis=0)

    a0 = np.median(pl_chain[:, 0])
    b0 = np.median(pl_chain[:, 1])

    ax.plot(log_p, med_model, 'r-', linewidth=2,
            label=f'M = {a0:.2f}(log P - 1) {"+" if b0 >= 0 else "-"} {abs(b0):.2f}')
    ax.fill_between(log_p, med_model - spread, med_model + spread,
                    color='gray', alpha=0.4, label=r'$1\sigma$')
    ax.fill_between(log_p, med_model - 2 * spread, med_model + 2 * spread,
                    color='gray', alpha=0.2, label=r'$2\sigma$')

    # --- Andromeda CV1 ---
    med_logp = np.median(np.log10(period_samples))
    logp_err = np.std(np.log10(period_samples))
    med_M = np.median(M_samples)
    M_err = np.std(M_samples)

    ax.scatter(np.log10(period_samples), M_samples,
               alpha=0.1, s=5, color='steelblue')
    ax.errorbar(med_logp, med_M, xerr=logp_err, yerr=M_err,
                fmt='D', color='steelblue', markersize=6,
                markeredgecolor='black', markeredgewidth=0.8,
                capsize=4, label='M31 CV1', zorder=6)

    ax.set_xlabel(r'$\log_{10}$(Period) [days]', fontsize=12)
    ax.set_ylabel('Absolute Magnitude', fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title('Period-Luminosity Relation with M31 CV1', fontsize=13)

    if savefig:
        fig.savefig('andromeda_pl_relation.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_light_curve(mjd, V_mag, V_err, period=None, savefig=False):
    """
    Publication-quality light curve. If period is given, also shows phased version.
    """
    if period is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Time series ---
    ax1.errorbar(mjd, V_mag, yerr=V_err, fmt='o', color='#4878A8',
                 markeredgecolor='black', markeredgewidth=0.2,
                 capsize=2, markersize=2, label='M31 CV1', zorder=3)
    ax1.set_xlabel('Time [MJD]', fontsize=12)
    ax1.set_ylabel('Apparent V Magnitude [mag]', fontsize=12)
    ax1.set_title('M31 CV1 Light Curve', fontsize=13)
    ax1.invert_yaxis()
    ax1.legend(fontsize=9, frameon=True, facecolor='white',
               edgecolor='black', framealpha=0.8)
    ax1.tick_params(which='both', direction='in', top=True, right=True)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(True, alpha=0.2, zorder=0)

    # --- Phased light curve ---
    if period is not None:
        phase = ((mjd - mjd[0]) % period) / period
        # plot two cycles for clarity
        for offset in [0, 1]:
            ax2.errorbar(phase + offset, V_mag, yerr=V_err, fmt='o',
                         color='#C44E52' if offset == 0 else '#C44E52',
                         markeredgecolor='black', markeredgewidth=0.6,
                         capsize=4, markersize=6, alpha=0.9 if offset == 0 else 0.35,
                         zorder=3)

        ax2.set_xlabel('Phase', fontsize=12)
        ax2.set_ylabel('Apparent V Magnitude [mag]', fontsize=12)
        ax2.set_title(f'Phased Light Curve (P = {period:.1f} d)', fontsize=13)
        ax2.invert_yaxis()
        ax2.set_xlim(-0.05, 2.05)
        ax2.tick_params(which='both', direction='in', top=True, right=True)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(True, alpha=0.2, zorder=0)

    plt.tight_layout()
    if savefig:
        fig.savefig('andromeda_light_curve.pdf', dpi=300, bbox_inches='tight')
    plt.show()