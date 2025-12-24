# =============================================================================
# Plotting Utilities
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os

from models.cib_analysis import (
    CIBData, CIBModel, CIBFitter,
    CIBDiagnostics, quick_fit
)

class CIBPlotter:
    """
    Plotting utilities for CIB analysis.

    All methods are static and can be called without instantiation.
    """

    @staticmethod
    def plot_bandpowers(data: CIBData, pairs: Optional[List[str]] = None,
                        physical_units: bool = True, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot bandpowers for selected frequency pairs.
        """
        if pairs is None:
            pairs = CIBData.SPIRE_PAIRS

        n_pairs = len(pairs)
        ncols = min(3, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, pair in zip(axes, pairs):
            ell, Cl, err = data.get_pair(pair, physical_units=physical_units)
            ax.errorbar(ell, Cl, yerr=err, fmt='o', markersize=4, capsize=2)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\ell$')
            ylabel = r'$C_\ell$ [MJy$^2$/sr]' if physical_units else r'$C_\ell$ [scaled]'
            ax.set_ylabel(ylabel)
            ax.set_title(f'{pair} GHz')
            ax.grid(True, alpha=0.3)

        for ax in axes[len(pairs):]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_fit(data: CIBData, model: CIBModel, params: np.ndarray,
                 pairs: Optional[List[str]] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot data with model fit.
        """
        if pairs is None:
            pairs = CIBData.SPIRE_PAIRS

        n_pairs = len(pairs)
        ncols = min(3, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        ell_plot = np.logspace(2.8, 4.1, 100)

        for ax, pair in zip(axes, pairs):
            f1, f2 = pair.split('x')
            nu1, nu2 = float(f1), float(f2)
            scale = data.scaling_factors[pair]

            ell, Cl, err = data.get_pair(pair, physical_units=False)
            ax.errorbar(ell, Cl, yerr=err, fmt='ko', markersize=4, capsize=2, label='Data')

            Cl_model = np.array([model.predict_scaled(l, nu1, nu2, params, scale)
                                 for l in ell_plot])
            ax.plot(ell_plot, Cl_model, 'r-', linewidth=2, label='Model')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'$C_\ell$ [scaled]')
            ax.set_title(f'{pair} GHz')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(500, 15000)

        for ax in axes[len(pairs):]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_residuals(data: CIBData, model: CIBModel, params: np.ndarray,
                       pairs: Optional[List[str]] = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot normalized residuals (data - model) / error.
        """
        if pairs is None:
            pairs = CIBData.SPIRE_PAIRS

        n_pairs = len(pairs)
        ncols = min(3, n_pairs)
        nrows = (n_pairs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, pair in zip(axes, pairs):
            f1, f2 = pair.split('x')
            nu1, nu2 = float(f1), float(f2)
            scale = data.scaling_factors[pair]

            ell, Cl, err = data.get_pair(pair, physical_units=False)
            Cl_model = np.array([model.predict_scaled(l, nu1, nu2, params, scale)
                                 for l in ell])

            residuals = (Cl - Cl_model) / err

            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(2, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(-2, color='gray', linestyle='--', alpha=0.5)
            ax.plot(ell, residuals, 'o-', markersize=5)

            ax.set_xscale('log')
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'(Data - Model) / $\sigma$')
            ax.set_title(f'{pair} GHz')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(500, 15000)
            ax.set_ylim(-5, 5)

        for ax in axes[len(pairs):]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def plot_correlation_coefficients(data: CIBData, figsize: Tuple[int, int] = (12, 5)):
        """
        Plot correlation coefficients r(ℓ) for cross-spectra.
        """
        diag = CIBDiagnostics(data)
        r_dict = diag.correlation_coefficients()
        ell = data.ell['600x600']

        fig, ax = plt.subplots(figsize=figsize)

        colors = {'600x857': 'blue', '600x1200': 'red', '857x1200': 'green'}
        markers = {'600x857': 'o', '600x1200': 's', '857x1200': '^'}

        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, linewidth=2)

        for pair, r in r_dict.items():
            label = f'{pair.replace("x", "×")} GHz (mean r={np.mean(r):.3f})'
            ax.plot(ell, r, marker=markers[pair], color=colors[pair],
                    markersize=6, linewidth=1.5, label=label)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\ell$', fontsize=12)
        ax.set_ylabel(r'Correlation coefficient $r(\ell)$', fontsize=12)
        ax.set_title('CIB Cross-Frequency Correlation Coefficients', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(500, 15000)
        ax.set_ylim(0.75, 1.05)

        plt.tight_layout()
        return fig, ax

