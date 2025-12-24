#!/usr/bin/env python3
"""
CIB Bandpower Analysis Module

Classes for loading, analyzing, and fitting CIB cross-frequency power spectra
from Viero et al. 2019 (ApJ 881:96) SPT×SPIRE data.

Classes:
    CIBData: Load and manage bandpower data, covariance matrices
    CIBModel: Define CIB halo models with various SED parameterizations
    CIBFitter: Fit models to data, compute chi-squared
    CIBDiagnostics: Compute correlation coefficients, SED ratios, etc.
    CIBPlotter: Visualization utilities

Example usage:
    >>> from cib_analysis import CIBData, CIBModel, CIBFitter, CIBPlotter
    >>> data = CIBData('/path/to/data/')
    >>> model = CIBModel(model_type='correlation')
    >>> fitter = CIBFitter(data, model)
    >>> result = fitter.fit()
    >>> CIBPlotter.plot_fit(data, model, result.x)

Author: Analysis of Viero et al. 2019 data
"""

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os


# =============================================================================
# Physical Constsants
# =============================================================================

H_PLANCK = 6.626e-34  # J·s
K_BOLTZ = 1.381e-23   # J/K
C_LIGHT = 3e8         # m/s


# =============================================================================
# Data Container
# =============================================================================

@dataclass
class FitResult:
    """Container for fit results"""
    params: np.ndarray
    chi2: float
    ndof: int
    param_names: List[str]
    success: bool
    
    @property
    def chi2_reduced(self) -> float:
        return self.chi2 / self.ndof
    
    def __repr__(self):
        lines = [f"FitResult(χ²/dof = {self.chi2:.1f}/{self.ndof} = {self.chi2_reduced:.2f})"]
        for name, val in zip(self.param_names, self.params):
            lines.append(f"  {name}: {val:.4f}")
        return "\n".join(lines)


# =============================================================================
# CIBData Class
# =============================================================================

class CIBData:
    """
    Load and manage CIB bandpower data from Viero et al. 2019.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing spectrum_sptxspire.txt, 
        covariance_sptxspire.txt, and info.txt
    
    Attributes
    ----------
    ell : dict
        Multipole values for each frequency pair
    bandpowers : dict
        Bandpower values in stored units (scaled)
    bandpowers_physical : dict
        Bandpower values in physical units (MJy²/sr)
    errors : dict
        Diagonal errors in stored units
    covariance : np.ndarray
        Full covariance matrix (scaled units)
    scaling_factors : dict
        Scaling factors applied to each pair
    unit_flags : dict
        Unit flags (0 = MJy²/sr, 1 = μK²_CMB)
    
    Examples
    --------
    >>> data = CIBData('/mnt/user-data/uploads/')
    >>> data.list_pairs()
    >>> ell, Cl, err = data.get_pair('600x600')
    """
    
    # Standard multipole bins for SPIRE (22 bins)
    ELL_SPIRE = np.array([700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2350, 2650, 
                          2950, 3300, 3700, 4150, 4650, 5200, 5849, 6599, 7399, 8299, 
                          9299, 10399])
    
    # Standard multipole bins for SPT (approximate - varies by pair)
    ELL_SPT = {
        12: np.array([2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000]),
        14: np.array([2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 9000]),
        15: np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]),
        20: np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 2800, 3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000]),
        21: np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 2800, 3200, 3600, 4000, 4500, 5000, 5500, 6000, 7000, 8000, 9000]),
        22: ELL_SPIRE  # SPIRE auto/cross
    }
    
    # All frequency pairs in order
    ALL_PAIRS = [
        '95x95', '95x150', '95x220', '150x150', '150x220', '220x220',
        '95x600', '95x857', '95x1200', '150x600', '150x857', '150x1200', 
        '220x600', '220x857', '220x1200',
        '600x600', '600x857', '600x1200', '857x857', '857x1200', '1200x1200'
    ]
    
    SPIRE_PAIRS = ['600x600', '600x857', '600x1200', '857x857', '857x1200', '1200x1200']
    SPT_PAIRS = ['95x95', '95x150', '95x220', '150x150', '150x220', '220x220']
    CROSS_PAIRS = ['95x600', '95x857', '95x1200', '150x600', '150x857', '150x1200', 
                   '220x600', '220x857', '220x1200']
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._load_data()
        
    def _get_ell_for_pair(self, n_bp: int) -> np.ndarray:
        """Get ell values for a pair with n_bp bandpowers"""
        if n_bp in self.ELL_SPT:
            return self.ELL_SPT[n_bp][:n_bp]
        else:
            # Fallback: generate reasonable ell values
            return np.logspace(np.log10(600), np.log10(10000), n_bp)
        
    def _load_data(self):
        """Load all data files"""
        # Load info file
        info_path = os.path.join(self.data_dir, 'info.txt')
        with open(info_path, 'r') as f:
            lines = f.readlines()
        
        # Parse number of bandpowers per pair
        self.n_bandpowers = {pair: int(lines[i].strip()) 
                            for i, pair in enumerate(self.ALL_PAIRS)}
        
        # Parse scaling factors and unit flags
        self.scaling_factors = {}
        self.unit_flags = {}
        for i, pair in enumerate(self.ALL_PAIRS):
            parts = lines[21 + i].strip().split()
            self.scaling_factors[pair] = float(parts[0])
            self.unit_flags[pair] = int(parts[1])
        
        # Load spectrum
        spectrum_path = os.path.join(self.data_dir, 'spectrum_sptxspire.txt')
        spectrum_raw = np.loadtxt(spectrum_path)
        
        # Load covariance
        cov_path = os.path.join(self.data_dir, 'covariance_sptxspire.txt')
        cov_raw = np.loadtxt(cov_path)
        n_total = int(np.sqrt(len(cov_raw)))
        self.covariance_full = cov_raw.reshape(n_total, n_total)
        
        # Parse into dictionaries
        self.ell = {}
        self.bandpowers = {}
        self.bandpowers_physical = {}
        self.errors = {}
        
        cumsum = np.cumsum([self.n_bandpowers[p] for p in self.ALL_PAIRS])
        starts = np.concatenate([[0], cumsum[:-1]])
        self._pair_indices = {pair: (starts[i], starts[i] + self.n_bandpowers[pair]) 
                              for i, pair in enumerate(self.ALL_PAIRS)}
        
        for pair in self.ALL_PAIRS:
            start, end = self._pair_indices[pair]
            n = self.n_bandpowers[pair]
            scale = self.scaling_factors[pair]
            
            # Use predefined ell values
            self.ell[pair] = self._get_ell_for_pair(n)
            self.bandpowers[pair] = spectrum_raw[start:end, 1]
            self.bandpowers_physical[pair] = self.bandpowers[pair] / scale
            
            # Diagonal errors from covariance
            self.errors[pair] = np.sqrt(np.diag(self.covariance_full[start:end, start:end]))
    
    def get_pair(self, pair: str, physical_units: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ell, bandpower, and error for a frequency pair.
        
        Parameters
        ----------
        pair : str
            Frequency pair, e.g., '600x600' or '600x1200'
        physical_units : bool
            If True, return values in MJy²/sr. If False, return scaled values.
            
        Returns
        -------
        ell, Cl, err : tuple of arrays
        """
        if pair not in self.ALL_PAIRS:
            raise ValueError(f"Unknown pair: {pair}. Use one of {self.ALL_PAIRS}")
        
        ell = self.ell[pair]
        if physical_units:
            Cl = self.bandpowers_physical[pair]
            err = self.errors[pair] / self.scaling_factors[pair]
        else:
            Cl = self.bandpowers[pair]
            err = self.errors[pair]
        
        return ell, Cl, err
    
    def get_covariance(self, pairs: List[str]) -> np.ndarray:
        """
        Extract covariance matrix for a subset of pairs.
        
        Parameters
        ----------
        pairs : list of str
            List of frequency pairs to include
            
        Returns
        -------
        cov : np.ndarray
            Covariance matrix for the selected pairs
        """
        indices = []
        for pair in pairs:
            start, end = self._pair_indices[pair]
            indices.extend(range(start, end))
        indices = np.array(indices)
        return self.covariance_full[np.ix_(indices, indices)]
    
    def get_inverse_covariance(self, pairs: List[str]) -> np.ndarray:
        """
        Get inverse covariance matrix via Cholesky decomposition.
        
        Parameters
        ----------
        pairs : list of str
            List of frequency pairs to include
            
        Returns
        -------
        cov_inv : np.ndarray
            Inverse covariance matrix
        """
        cov = self.get_covariance(pairs)
        L = linalg.cholesky(cov, lower=True)
        return linalg.cho_solve((L, True), np.eye(len(cov)))
    
    def build_data_vectors(self, pairs: List[str]) -> Tuple[np.ndarray, ...]:
        """
        Build flattened data vectors for fitting.
        
        Parameters
        ----------
        pairs : list of str
            List of frequency pairs to include
            
        Returns
        -------
        ell_data, Cl_data, err_data, scale_data, nu1_data, nu2_data : tuple of arrays
        """
        ell_data = []
        Cl_data = []
        err_data = []
        scale_data = []
        nu1_data = []
        nu2_data = []
        
        for pair in pairs:
            f1, f2 = pair.split('x')
            nu1, nu2 = float(f1), float(f2)
            ell, Cl, err = self.get_pair(pair, physical_units=False)
            scale = self.scaling_factors[pair]
            
            for i in range(len(ell)):
                ell_data.append(ell[i])
                Cl_data.append(Cl[i])
                err_data.append(err[i])
                scale_data.append(scale)
                nu1_data.append(nu1)
                nu2_data.append(nu2)
        
        return (np.array(ell_data), np.array(Cl_data), np.array(err_data),
                np.array(scale_data), np.array(nu1_data), np.array(nu2_data))
    
    def list_pairs(self):
        """Print available frequency pairs with info"""
        print(f"{'Pair':<12} {'N_bp':<6} {'Scale':<10} {'Units':<12}")
        print("-" * 40)
        for pair in self.ALL_PAIRS:
            n = self.n_bandpowers[pair]
            scale = self.scaling_factors[pair]
            unit = "μK²_CMB" if self.unit_flags[pair] == 1 else "MJy²/sr"
            print(f"{pair:<12} {n:<6} {scale:<10.4f} {unit:<12}")


# =============================================================================
# CIB Models
# =============================================================================

class CIBModel:
    """
    CIB power spectrum models with various SED parameterizations.
    
    Parameters
    ----------
    model_type : str
        One of:
        - 'simple': Free SED amplitudes s_857, s_1200 (assumes r=1)
        - 'correlation': Free SED amplitudes and correlation coefficients
        
    n_terms : int
        Number of power spectrum terms (1, 2, or 3)
        - 1: Single amplitude
        - 2: Shot noise + clustering (power law)
        - 3: Shot noise + 1-halo + 2-halo terms
        
    Attributes
    ----------
    param_names : list
        Names of model parameters
    n_params : int
        Number of parameters
    bounds : list of tuples
        Parameter bounds for optimization
        
    Examples
    --------
    >>> model = CIBModel('correlation', n_terms=3)
    >>> Cl = model.predict(ell=3000, nu1=600, nu2=857, params=best_fit)
    """
    
    def __init__(self, model_type: str = 'simple', n_terms: int = 2):
        self.model_type = model_type
        self.n_terms = n_terms
        self._setup_model()
        
    def _setup_model(self):
        """Set up parameter names and bounds based on model type"""
        
        # Power spectrum parameters
        if self.n_terms == 1:
            self._ps_params = ['A']
            self._ps_bounds = [(0, None)]
        elif self.n_terms == 2:
            self._ps_params = ['C_shot', 'C_clust', 'alpha_clust']
            self._ps_bounds = [(0, None), (0, None), (-3, 1)]
        elif self.n_terms == 3:
            self._ps_params = ['C_shot', 'C_1h', 'alpha_1h', 'C_2h', 'alpha_2h']
            self._ps_bounds = [(0, None), (0, None), (-2, 1), (0, None), (-3, 0)]
        else:
            raise ValueError(f"n_terms must be 1, 2, or 3, got {self.n_terms}")
        
        # SED parameters
        if self.model_type == 'simple':
            self._sed_params = ['s857', 's1200']
            self._sed_bounds = [(0, None), (0, None)]
            self._predict_func = self._predict_simple
            
        elif self.model_type == 'correlation':
            self._sed_params = ['A857', 'A1200', 'r_600_857', 'r_600_1200', 'r_857_1200']
            self._sed_bounds = [(0, None), (0, None), (0, 1), (0, 1), (0, 1)]
            self._predict_func = self._predict_correlation
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.param_names = self._ps_params + self._sed_params
        self.n_params = len(self.param_names)
        self.bounds = self._ps_bounds + self._sed_bounds
        
    def _power_spectrum(self, ell: float, ps_params: np.ndarray) -> float:
        """Compute power spectrum shape P(ell)"""
        ell_pivot = 3000.
        
        # Ensure ell is positive
        if ell <= 0:
            return 0.0
        
        if self.n_terms == 1:
            return ps_params[0]
        elif self.n_terms == 2:
            C_shot, C_clust, alpha = ps_params
            return C_shot + C_clust * (ell / ell_pivot)**alpha
        elif self.n_terms == 3:
            C_shot, C_1h, alpha_1h, C_2h, alpha_2h = ps_params
            return C_shot + C_1h * (ell / ell_pivot)**alpha_1h + C_2h * (ell / ell_pivot)**alpha_2h
    
    def _predict_simple(self, ell: float, nu1: float, nu2: float, params: np.ndarray) -> float:
        """Simple model: C_ij = s_i × s_j × P(ell)"""
        n_ps = len(self._ps_params)
        ps_params = params[:n_ps]
        s857, s1200 = params[n_ps:]
        
        sed = {600.: 1.0, 857.: s857, 1200.: s1200}
        sed_factor = sed[nu1] * sed[nu2]
        
        return sed_factor * self._power_spectrum(ell, ps_params)
    
    def _predict_correlation(self, ell: float, nu1: float, nu2: float, params: np.ndarray) -> float:
        """Correlation model: C_ij = r_ij × sqrt(A_i × A_j) × P(ell)"""
        n_ps = len(self._ps_params)
        ps_params = params[:n_ps]
        A857, A1200, r_600_857, r_600_1200, r_857_1200 = params[n_ps:]
        
        # Amplitude dictionary (A_600 = 1 by definition)
        A = {600.: 1.0, 857.: A857, 1200.: A1200}
        
        # Correlation dictionary
        r = {
            (600., 600.): 1.0, (857., 857.): 1.0, (1200., 1200.): 1.0,
            (600., 857.): r_600_857, (857., 600.): r_600_857,
            (600., 1200.): r_600_1200, (1200., 600.): r_600_1200,
            (857., 1200.): r_857_1200, (1200., 857.): r_857_1200,
        }
        
        return r[(nu1, nu2)] * np.sqrt(A[nu1] * A[nu2]) * self._power_spectrum(ell, ps_params)
    
    def predict(self, ell: float, nu1: float, nu2: float, params: np.ndarray) -> float:
        """
        Predict power spectrum in physical units (MJy²/sr at 600×600 reference).
        
        Parameters
        ----------
        ell : float
            Multipole
        nu1, nu2 : float
            Frequencies in GHz
        params : array
            Model parameters
            
        Returns
        -------
        Cl : float
            Power spectrum in MJy²/sr
        """
        return self._predict_func(ell, nu1, nu2, params)
    
    def predict_scaled(self, ell: float, nu1: float, nu2: float, 
                       params: np.ndarray, scale: float) -> float:
        """Predict power spectrum in scaled (stored) units"""
        return self.predict(ell, nu1, nu2, params) * scale
    
    def get_default_params(self) -> np.ndarray:
        """Get reasonable starting parameters for optimization"""
        if self.n_terms == 2:
            ps_defaults = [2000., 1500., -1.2]
        elif self.n_terms == 3:
            ps_defaults = [0., 3000., -0.15, 1000., -1.7]
        else:
            ps_defaults = [3000.]
        
        if self.model_type == 'simple':
            sed_defaults = [1.55, 1.95]
        elif self.model_type == 'correlation':
            sed_defaults = [2.4, 3.8, 0.97, 0.86, 0.955]
        
        return np.array(ps_defaults + sed_defaults)
    
    def check_bounds(self, params: np.ndarray) -> bool:
        """Check if parameters are within bounds"""
        for i, (val, (lo, hi)) in enumerate(zip(params, self.bounds)):
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
        return True


# =============================================================================
# CIB Fitter
# =============================================================================

class CIBFitter:
    """
    Fit CIB models to bandpower data.
    
    Parameters
    ----------
    data : CIBData
        Data object
    model : CIBModel
        Model object
    pairs : list of str, optional
        Frequency pairs to fit. Default is SPIRE-only.
    use_full_covariance : bool
        If True, use full covariance matrix. If False, use diagonal only.
        
    Examples
    --------
    >>> data = CIBData('/path/to/data/')
    >>> model = CIBModel('correlation', n_terms=3)
    >>> fitter = CIBFitter(data, model)
    >>> result = fitter.fit()
    >>> print(result)
    """
    
    def __init__(self, data: CIBData, model: CIBModel, 
                 pairs: Optional[List[str]] = None,
                 use_full_covariance: bool = False):
        self.data = data
        self.model = model
        self.pairs = pairs or CIBData.SPIRE_PAIRS
        self.use_full_covariance = use_full_covariance
        
        # Build data vectors
        (self.ell_data, self.Cl_data, self.err_data, 
         self.scale_data, self.nu1_data, self.nu2_data) = data.build_data_vectors(self.pairs)
        
        self.n_data = len(self.ell_data)
        
        # Get inverse covariance if needed
        if use_full_covariance:
            self.cov_inv = data.get_inverse_covariance(self.pairs)
        else:
            self.cov_inv = None
    
    def chi2(self, params: np.ndarray) -> float:
        """Compute chi-squared for given parameters"""
        if not self.model.check_bounds(params):
            return 1e20
        
        # Compute model predictions
        try:
            model_pred = np.array([
                self.model.predict_scaled(self.ell_data[i], self.nu1_data[i], 
                                          self.nu2_data[i], params, self.scale_data[i])
                for i in range(self.n_data)
            ])
        except (ValueError, FloatingPointError):
            return 1e20
        
        # Check for invalid model predictions
        if not np.all(np.isfinite(model_pred)):
            return 1e20
        
        residuals = self.Cl_data - model_pred
        
        if self.use_full_covariance:
            chi2_val = residuals @ self.cov_inv @ residuals
        else:
            chi2_val = np.sum((residuals / self.err_data)**2)
        
        return chi2_val if np.isfinite(chi2_val) else 1e20
    
    def fit(self, p0: Optional[np.ndarray] = None, 
            method: str = 'Nelder-Mead', 
            maxiter: int = 50000,
            two_step: bool = True) -> FitResult:
        """
        Fit model to data.
        
        Parameters
        ----------
        p0 : array, optional
            Starting parameters. Default uses model defaults.
        method : str
            Optimization method for scipy.optimize.minimize
        maxiter : int
            Maximum iterations
        two_step : bool
            If True, do a coarse fit first then refine
            
        Returns
        -------
        result : FitResult
            Fit results including best-fit parameters and chi-squared
        """
        if p0 is None:
            p0 = self.model.get_default_params()
        
        if two_step:
            # First pass: coarse optimization
            result1 = minimize(self.chi2, p0, method='Nelder-Mead', 
                             options={'maxiter': maxiter // 2, 'xatol': 1e-6, 'fatol': 1e-6})
            
            # Second pass: refine from first result
            result = minimize(self.chi2, result1.x, method='Nelder-Mead', 
                             options={'maxiter': maxiter // 2, 'xatol': 1e-8, 'fatol': 1e-8})
        else:
            result = minimize(self.chi2, p0, method=method, 
                             options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
        
        ndof = self.n_data - self.model.n_params
        
        return FitResult(
            params=result.x,
            chi2=result.fun,
            ndof=ndof,
            param_names=self.model.param_names,
            success=result.success
        )
    
    def chi2_by_pair(self, params: np.ndarray) -> Dict[str, Tuple[float, int]]:
        """
        Compute chi-squared for each frequency pair separately.
        
        Parameters
        ----------
        params : array
            Model parameters
            
        Returns
        -------
        chi2_dict : dict
            Dictionary mapping pair name to (chi2, n_points)
        """
        chi2_dict = {}
        
        for pair in self.pairs:
            f1, f2 = pair.split('x')
            nu1, nu2 = float(f1), float(f2)
            
            mask = (self.nu1_data == nu1) & (self.nu2_data == nu2)
            ell_pair = self.ell_data[mask]
            Cl_pair = self.Cl_data[mask]
            err_pair = self.err_data[mask]
            scale_pair = self.scale_data[mask]
            
            model_pred = np.array([
                self.model.predict_scaled(ell_pair[i], nu1, nu2, params, scale_pair[i])
                for i in range(len(ell_pair))
            ])
            
            chi2 = np.sum(((Cl_pair - model_pred) / err_pair)**2)
            chi2_dict[pair] = (chi2, len(ell_pair))
        
        return chi2_dict


# =============================================================================
# Diagnostics
# =============================================================================

class CIBDiagnostics:
    """
    Diagnostic tools for CIB analysis.
    
    Parameters
    ----------
    data : CIBData
        Data object
        
    Examples
    --------
    >>> data = CIBData('/path/to/data/')
    >>> diag = CIBDiagnostics(data)
    >>> r = diag.correlation_coefficients()
    >>> ratios = diag.sed_ratios()
    """
    
    def __init__(self, data: CIBData):
        self.data = data
    
    def correlation_coefficients(self, pairs: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Compute Pearson correlation coefficients r(ℓ) for cross-spectra.
        
        r(ℓ) = C_ij(ℓ) / sqrt(C_ii(ℓ) × C_jj(ℓ))
        
        Parameters
        ----------
        pairs : list of str, optional
            Cross-spectrum pairs to compute. Default is SPIRE cross-spectra.
            
        Returns
        -------
        r_dict : dict
            Dictionary mapping pair name to array of r values
        """
        if pairs is None:
            pairs = ['600x857', '600x1200', '857x1200']
        
        r_dict = {}
        
        for pair in pairs:
            f1, f2 = pair.split('x')
            if f1 == f2:
                continue  # Skip auto-spectra
            
            auto1 = f'{f1}x{f1}'
            auto2 = f'{f2}x{f2}'
            
            _, C_cross, _ = self.data.get_pair(pair, physical_units=True)
            _, C_auto1, _ = self.data.get_pair(auto1, physical_units=True)
            _, C_auto2, _ = self.data.get_pair(auto2, physical_units=True)
            
            r = C_cross / np.sqrt(C_auto1 * C_auto2)
            r_dict[pair] = r
        
        return r_dict
    
    def correlation_summary(self) -> None:
        """Print summary of correlation coefficients"""
        r_dict = self.correlation_coefficients()
        ell = self.data.ell['600x600']
        
        print("="*60)
        print("CORRELATION COEFFICIENTS r(ℓ) = C_ij / sqrt(C_ii × C_jj)")
        print("="*60)
        
        for pair, r in r_dict.items():
            print(f"\n{pair}:")
            print(f"  Mean r = {np.mean(r):.4f} ± {np.std(r)/np.sqrt(len(r)):.4f}")
            print(f"  Range: [{np.min(r):.4f}, {np.max(r):.4f}]")
        
        # Binned summary
        print("\n" + "-"*60)
        print("Binned averages:")
        print(f"{'Pair':<12} {'ℓ<2000':<10} {'2000<ℓ<5000':<14} {'ℓ>5000':<10}")
        
        for pair, r in r_dict.items():
            low_mask = ell < 2000
            mid_mask = (ell >= 2000) & (ell < 5000)
            high_mask = ell >= 5000
            
            low = np.mean(r[low_mask]) if np.any(low_mask) else np.nan
            mid = np.mean(r[mid_mask]) if np.any(mid_mask) else np.nan
            high = np.mean(r[high_mask]) if np.any(high_mask) else np.nan
            print(f"{pair:<12} {low:<10.4f} {mid:<14.4f} {high:<10.4f}")
    
    def sed_ratios(self, ell_range: Tuple[float, float] = (5000, 12000)) -> Dict[str, float]:
        """
        Compute SED ratios from high-ℓ bandpowers.
        
        Parameters
        ----------
        ell_range : tuple
            Range of ℓ to average over (should be Poisson-dominated)
            
        Returns
        -------
        ratios : dict
            Dictionary mapping pair to ratio relative to 600×600
        """
        ell = self.data.ell['600x600']
        mask = (ell >= ell_range[0]) & (ell <= ell_range[1])
        
        _, Cl_ref, _ = self.data.get_pair('600x600', physical_units=True)
        ref = np.mean(Cl_ref[mask])
        
        ratios = {}
        for pair in CIBData.SPIRE_PAIRS:
            _, Cl, _ = self.data.get_pair(pair, physical_units=True)
            ratios[pair] = np.mean(Cl[mask]) / ref
        
        return ratios
    
    def check_covariance(self, pairs: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Check covariance matrix properties.
        
        Returns condition number, Cholesky success, etc.
        """
        if pairs is None:
            pairs = CIBData.SPIRE_PAIRS
        
        cov = self.data.get_covariance(pairs)
        
        results = {}
        results['condition_number'] = np.linalg.cond(cov)
        results['min_eigenvalue'] = np.min(np.linalg.eigvalsh(cov))
        results['max_eigenvalue'] = np.max(np.linalg.eigvalsh(cov))
        
        try:
            L = linalg.cholesky(cov, lower=True)
            results['cholesky_success'] = True
            
            # Check inversion quality
            cov_inv = linalg.cho_solve((L, True), np.eye(len(cov)))
            results['inversion_error'] = np.max(np.abs(cov @ cov_inv - np.eye(len(cov))))
        except linalg.LinAlgError:
            results['cholesky_success'] = False
            results['inversion_error'] = np.inf
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_fit(data_dir: str, model_type: str = 'correlation', 
              n_terms: int = 3, use_full_cov: bool = False) -> Tuple[FitResult, CIBFitter]:
    """
    Quick fit with sensible defaults.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory
    model_type : str
        Model type ('simple' or 'correlation')
    n_terms : int
        Number of power spectrum terms
    use_full_cov : bool
        Whether to use full covariance matrix
        
    Returns
    -------
    result : FitResult
        Fit results
    fitter : CIBFitter
        Fitter object for further analysis
    """
    data = CIBData(data_dir)
    model = CIBModel(model_type=model_type, n_terms=n_terms)
    fitter = CIBFitter(data, model, use_full_covariance=use_full_cov)
    result = fitter.fit()
    return result, fitter


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == '__main__':
    # Quick test
    data_dir = '/mnt/user-data/uploads'
    
    print("Loading data...")
    data = CIBData(data_dir)
    data.list_pairs()
    
    print(f"\nSPIRE ell values: {data.ell['600x600']}")
    
    print("\n" + "="*60)
    print("Diagnostics...")
    diag = CIBDiagnostics(data)
    diag.correlation_summary()
    
    print("\n" + "="*60)
    print("Fitting simple model (3-term)...")
    model_simple = CIBModel('simple', n_terms=3)
    
    # Use good starting params from earlier analysis
    p0_simple = np.array([0., 3000., -0.13, 1000., -1.76, 1.57, 1.93])
    
    fitter_simple = CIBFitter(data, model_simple)
    result_simple = fitter_simple.fit(p0=p0_simple)
    print(result_simple)
    
    print("\n" + "="*60)
    print("Fitting correlation model (3-term)...")
    model_corr = CIBModel('correlation', n_terms=3)
    
    # Use good starting params from earlier analysis
    p0_corr = np.array([0., 3087., -0.136, 1020., -1.765, 2.425, 3.775, 0.970, 0.860, 0.955])
    
    fitter_corr = CIBFitter(data, model_corr)
    result_corr = fitter_corr.fit(p0=p0_corr)
    print(result_corr)
    
    print("\n" + "="*60)
    print("Chi² by pair (correlation model):")
    chi2_by_pair = fitter_corr.chi2_by_pair(result_corr.params)
    for pair, (chi2, n) in chi2_by_pair.items():
        print(f"  {pair}: χ²={chi2:.1f}/{n} = {chi2/n:.2f}")
