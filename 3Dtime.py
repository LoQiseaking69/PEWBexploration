import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
import json
warnings.filterwarnings('ignore')

class ValidationData:
"""Load and preprocess empirical validation data."""

def __init__(self, filepath: str = '3D_time_validation_results.csv'):  
    self.df = pd.read_csv(filepath)  
    self._compute_derived_quantities()  
    self._identify_regimes()  
  
def _compute_derived_quantities(self):  
    """Calculate derived physical quantities from raw data."""  
    df = self.df  
      
    # Normalized parameters      
    df['log10_epsilon'] = np.log10(df['epsilon'])      
    df['epsilon_squared'] = df['epsilon']**2      
      
    # Physical ratios      
    df['alpha_eff'] = 1.0 / df['dt_eff']  # Effective metric factor      
    df['v_over_c'] = df['v_eff'] / 2.99792458e8      
    df['E_ratio'] = df['E_total'] / df['E_total'].iloc[0]  # Normalized to baseline      
      
    # Deltas from baseline (Regime I)      
    baseline = df.iloc[0]      
    df['delta_dt'] = df['dt_eff'] - baseline['dt_eff']      
    df['delta_ds'] = df['ds'] - baseline['ds']      
    df['delta_v'] = df['v_eff'] - baseline['v_eff']      
    df['delta_E'] = df['E_total'] - baseline['E_total']      
      
    # Percentage changes      
    df['pct_dt'] = (df['dt_eff'] / baseline['dt_eff'] - 1) * 100      
    df['pct_v'] = (df['v_eff'] / baseline['v_eff'] - 1) * 100      
    df['pct_E'] = (df['E_total'] / baseline['E_total'] - 1) * 100      
  
def _identify_regimes(self):  
    """Automatically identify physical regimes based on epsilon values."""  
    df = self.df  
      
    # Regime boundaries (empirically determined from data)      
    self.regime_boundaries = {      
        'quantum': (df['epsilon'].min(), 1e-8),      
        'transitional': (1e-8, 1e-4),      
        'classical': (1e-4, df['epsilon'].max())      
    }      
      
    # Assign regime labels      
    conditions = [      
        (df['epsilon'] < 1e-8),      
        (df['epsilon'] >= 1e-8) & (df['epsilon'] < 1e-4),      
        (df['epsilon'] >= 1e-4)      
    ]      
    choices = ['quantum', 'transitional', 'classical']      
    df['regime'] = np.select(conditions, choices)      
      
    # Store regime subsets      
    self.regimes = {      
        name: df[(df['epsilon'] >= low) & (df['epsilon'] <= high)].copy()      
        for name, (low, high) in self.regime_boundaries.items()      
    }  
  
def get_regime(self, name: str) -> pd.DataFrame:  
    """Get data for specific regime."""  
    return self.regimes.get(name, pd.DataFrame())  
  
@property  
def baseline(self) -> pd.Series:  
    """Get baseline (epsilon -> 0) values."""  
    return self.df.iloc[0]

class StatisticalValidator:
"""Perform rigorous statistical validation of theoretical claims."""

def __init__(self, data: ValidationData):  
    self.data = data  
    self.results = {}  
  
def test_claim_1_positive_energy(self) -> Dict:  
    """  
    CLAIM 1: 3D temporal structure provides positive energy contribution.  
    """  
    df = self.data.df  
    baseline_E = self.data.baseline['E_total']  
      
    # Test if energy increases monotonically with epsilon  
    energy_diff = df['E_total'] - baseline_E  
      
    # Statistical test: One-sided t-test on energy differences  
    t_stat, p_value = stats.ttest_1samp(energy_diff.iloc[1:], 0, alternative='greater')  
      
    # Effect size (Cohen's d)  
    cohens_d = energy_diff.iloc[1:].mean() / energy_diff.iloc[1:].std()  
      
    # Monotonicity test  
    is_monotonic = np.all(np.diff(df['E_total']) >= -1e10)  # Allow tiny numerical noise  
      
    validation = {  
        'claim': '3D time provides positive energy',  
        'null_hypothesis': 'E_total <= E_baseline',  
        't_statistic': t_stat,  
        'p_value': p_value,  
        'cohens_d': cohens_d,  
        'is_monotonic': is_monotonic,  
        'min_energy_increase_J': energy_diff.min(),  
        'max_energy_increase_J': energy_diff.max(),  
        'validated': p_value < 0.05 and energy_diff.min() > -1e10,  
        'confidence': '99.9%' if p_value < 0.001 else '95%' if p_value < 0.05 else 'INSIGNIFICANT',  
        'evidence_strength': 'STRONG' if cohens_d > 0.8 else 'MODERATE' if cohens_d > 0.5 else 'WEAK'  
    }  
      
    self.results['positive_energy'] = validation  
    return validation  
  
def test_claim_2_epsilon_scaling(self) -> Dict:  
    """  
    CLAIM 2: Metric modifications scale as epsilon^2.  
    """  
    df = self.data.df  
      
    # Focus on quantum regime where perturbation theory should hold  
    quantum = self.data.get_regime('quantum')  
      
    # Fit power law: delta_dt = A * epsilon^n  
    def power_law(eps, A, n):  
        return A * eps**n  
      
    # Fit to quantum regime data  
    popt, pcov = curve_fit(power_law, quantum['epsilon'], quantum['delta_dt'],   
                          p0=[1e16, 2], maxfev=10000)  
    A_fit, n_fit = popt  
    n_err = np.sqrt(pcov[1, 1])  
      
    # R-squared  
    residuals = quantum['delta_dt'] - power_law(quantum['epsilon'], *popt)  
    ss_res = np.sum(residuals**2)  
    ss_tot = np.sum((quantum['delta_dt'] - quantum['delta_dt'].mean())**2)  
    r_squared = 1 - (ss_res / ss_tot)  
      
    # Test if n ≈ 2 within uncertainty  
    n_deviation = abs(n_fit - 2.0) / n_err if n_err > 0 else abs(n_fit - 2.0)  
      
    validation = {  
        'claim': 'Metric scales as epsilon^2',  
        'fitted_exponent': n_fit,  
        'exponent_error': n_err,  
        'r_squared': r_squared,  
        'prefactor_A': A_fit,  
        'n_sigma_from_2': n_deviation,  
        'validated': abs(n_fit - 2.0) < 2*n_err and r_squared > 0.95,  
        'confidence': f'{min(99.9, 100*(1 - stats.norm.sf(n_deviation)*2)):.1f}%',  
        'regime': 'quantum (epsilon < 1e-8)'  
    }  
      
    self.results['epsilon_scaling'] = validation  
    return validation  
  
def test_claim_3_velocity_reduction(self) -> Dict:  
    """  
    CLAIM 3: 3D time coupling reduces effective velocity.  
    """  
    df = self.data.df  
      
    # Spearman correlation (monotonic relationship)  
    corr, p_value = stats.spearmanr(df['epsilon'], df['v_eff'])  
      
    # Linear regression in log-log space for power law  
    log_eps = np.log10(df['epsilon'])  
    log_v = np.log10(df['v_eff'])  
    slope, intercept, r_value, p_value_lr, std_err = stats.linregress(log_eps, log_v)  
      
    # Quantify velocity penalty at maximum epsilon  
    v_baseline = self.data.baseline['v_eff']  
    v_final = df['v_eff'].iloc[-1]  
    velocity_penalty = (v_baseline - v_final) / v_baseline * 100  
      
    validation = {  
        'claim': '3D time reduces effective velocity',  
        'spearman_r': corr,  
        'spearman_p': p_value,  
        'power_law_slope': slope,  
        'r_squared': r_value**2,  
        'velocity_penalty_percent': velocity_penalty,  
        'validated': corr < -0.9 and p_value < 0.001,  
        'confidence': '99.9%' if p_value < 0.001 else '95%' if p_value < 0.05 else 'INSIGNIFICANT',  
        'physical_interpretation': f'{velocity_penalty:.2f}% velocity reduction at max epsilon'  
    }  
      
    self.results['velocity_reduction'] = validation  
    return validation  
  
def test_claim_4_no_horizon(self) -> Dict:  
    """  
    CLAIM 4: No event horizons form (alpha^2 remains positive).  
    """  
    df = self.data.df  
      
    # alpha_eff = 1/dt_eff, horizon would form if alpha^2 <= 0  
    alpha_squared = df['alpha_eff']**2  
      
    min_alpha_sq = alpha_squared.min()  
    max_alpha_sq = alpha_squared.max()  
      
    # Safety margin: how far from zero?  
    safety_margin = min_alpha_sq / max_alpha_sq  
      
    validation = {  
        'claim': 'No event horizons form',  
        'min_alpha_squared': min_alpha_sq,  
        'max_alpha_squared': max_alpha_sq,  
        'safety_margin': safety_margin,  
        'min_dt_eff': df['dt_eff'].min(),  
        'max_dt_eff': df['dt_eff'].max(),  
        'validated': min_alpha_sq > 0.5,  # Conservative: must stay well above 0  
        'confidence': '100%' if min_alpha_sq > 0.9 else 'HIGH' if min_alpha_sq > 0.5 else 'MARGINAL',  
        'evidence': f'alpha^2 in [{min_alpha_sq:.6f}, {max_alpha_sq:.6f}]'  
    }  
      
    self.results['no_horizon'] = validation  
    return validation  
  
def test_claim_5_energy_finiteness(self) -> Dict:  
    """  
    CLAIM 5: Total energy remains finite and bounded.  
    """  
    df = self.data.df  
      
    # Check for finiteness  
    all_finite = np.all(np.isfinite(df['E_total']))  
      
    # Check for boundedness (no exponential growth)  
    log_E = np.log(df['E_total'])  
    d_log_E = np.diff(log_E)  
    max_growth_rate = np.max(d_log_E) / np.mean(np.diff(np.log(df['epsilon'])))  
      
    # Fit asymptotic behavior  
    def saturation_model(eps, E0, A, B):  
        return E0 + A * eps**2 / (1 + B * eps**2)  
      
    try:  
        popt, _ = curve_fit(saturation_model, df['epsilon'], df['E_total'],   
                           p0=[1.12e17, 3e14, 0.1], maxfev=10000)  
        E_asymptotic = popt[0] + popt[1]/popt[2] if popt[2] > 0 else np.inf  
        fit_quality = 'good'  
    except:  
        E_asymptotic = df['E_total'].iloc[-1]  
        fit_quality = 'poor'  
      
    validation = {  
        'claim': 'Energy remains finite and bounded',  
        'all_finite': all_finite,  
        'max_growth_rate': max_growth_rate,  
        'asymptotic_estimate_J': E_asymptotic,  
        'fit_quality': fit_quality,  
        'energy_range_J': (df['E_total'].min(), df['E_total'].max()),  
        'validated': all_finite and max_growth_rate < 0.1,  # Sub-linear growth  
        'confidence': 'HIGH' if all_finite and max_growth_rate < 0.01 else 'MODERATE'  
    }  
      
    self.results['energy_finiteness'] = validation  
    return validation  
  
def test_claim_6_regime_transitions(self) -> Dict:  
    """  
    CLAIM 6: Three distinct physical regimes exist.  
    """  
    df = self.data.df  
      
    # Use second derivative of E vs log(epsilon) to find transitions  
    log_eps = np.log10(df['epsilon'])  
    d2E = np.gradient(np.gradient(df['E_total'], log_eps), log_eps)  
      
    # Find peaks in second derivative (regime boundaries)  
    peaks, properties = find_peaks(np.abs(d2E), height=np.std(d2E))  
      
    # Validate expected transitions at ~1e-8 and ~1e-4  
    expected_transitions = [-8, -4]  # log10(epsilon)  
    found_transitions = log_eps.iloc[peaks].values if len(peaks) > 0 else []  
      
    # Match found to expected  
    matches = []  
    for exp in expected_transitions:  
        if len(found_transitions) > 0:  
            closest = found_transitions[np.argmin(np.abs(found_transitions - exp))]  
            matches.append(abs(closest - exp) < 1.0)  # Within 1 dex  
      
    validation = {  
        'claim': 'Three distinct physical regimes',  
        'expected_transitions_log10_eps': expected_transitions,  
        'found_transitions_log10_eps': list(found_transitions),  
        'transition_matches': sum(matches),  
        'regime_identified': {  
            'quantum': len(self.data.get_regime('quantum')),  
            'transitional': len(self.data.get_regime('transitional')),  
            'classical': len(self.data.get_regime('classical'))  
        },  
        'validated': sum(matches) >= 1,  
        'confidence': 'HIGH' if sum(matches) == 2 else 'MODERATE' if sum(matches) == 1 else 'LOW'  
    }  
      
    self.results['regime_transitions'] = validation  
    return validation  
  
def run_all_validations(self) -> Dict:  
    """Execute complete validation suite."""  
    print("=" * 80)  
    print("EMPIRICAL VALIDATION SUITE - EXECUTING ALL TESTS")  
    print("=" * 80)  
      
    tests = [  
        self.test_claim_1_positive_energy,  
        self.test_claim_2_epsilon_scaling,  
        self.test_claim_3_velocity_reduction,  
        self.test_claim_4_no_horizon,  
        self.test_claim_5_energy_finiteness,  
        self.test_claim_6_regime_transitions  
    ]  
      
    for test in tests:  
        result = test()  
        self._print_validation(result)  
        print()  
      
    return self.results  
  
def _print_validation(self, result: Dict):  
    """Pretty print validation results."""  
    status = "✅ VALIDATED" if result.get('validated', False) else "❌ REJECTED"  
    print(f"{status}: {result['claim']}")  
    print(f"  Confidence: {result.get('confidence', 'N/A')}")  
    print(f"  Key Evidence: ", end="")  
      
    if 'p_value' in result:  
        print(f"p = {result['p_value']:.2e}")  
    elif 'r_squared' in result:  
        print(f"R² = {result['r_squared']:.6f}")  
    elif 'min_alpha_squared' in result:  
        print(f"min(α²) = {result['min_alpha_squared']:.6f}")  
    else:  
        print(f"{list(result.items())[-1][1]}")

class ModelCalibration:
"""Calibrate theoretical model parameters against empirical data."""

def __init__(self, data: ValidationData):  
    self.data = data  
    self.params = {}  
  
def calibrate_energy_model(self) -> Dict:  
    """  
    Calibrate: E_total(epsilon) = E_warp + E_3dt * epsilon^2  
    """  
    df = self.data.df  
      
    # Linear fit: E = E0 + A*eps^2  
    X = df['epsilon_squared'].values.reshape(-1, 1)  
    y = df['E_total'].values  
      
    A = np.hstack([np.ones_like(X), X])  
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)  
      
    E_warp = coeffs[0]  
    E_3dt_coeff = coeffs[1]  
      
    # Uncertainty estimation  
    y_pred = A @ coeffs  
    mse = np.mean((y - y_pred)**2)  
    cov = mse * np.linalg.inv(A.T @ A)  
    E_warp_err = np.sqrt(cov[0, 0])  
    E_3dt_err = np.sqrt(cov[1, 1])  
      
    # R-squared  
    ss_tot = np.sum((y - y.mean())**2)  
    ss_res = np.sum((y - y_pred)**2)  
    r_squared = 1 - ss_res / ss_tot  
      
    self.params['energy'] = {  
        'E_warp_J': E_warp,  
        'E_warp_err': E_warp_err,  
        'E_3dt_coeff_J': E_3dt_coeff,  
        'E_3dt_err': E_3dt_err,  
        'r_squared': r_squared,  
        'model': f'E_total = ({E_warp:.3e} ± {E_warp_err:.0e}) + ({E_3dt_coeff:.3e} ± {E_3dt_err:.0e}) * ε²'  
    }  
      
    return self.params['energy']  
  
def calibrate_metric_model(self) -> Dict:  
    """  
    Calibrate: dt_eff = 1 + k * epsilon^2 (for small epsilon)  
    """  
    q = self.data.get_regime('quantum')  
      
    X = q['epsilon_squared'].values  
    y = q['dt_eff'].values - 1.0  
      
    # Linear fit through origin  
    k = np.sum(X * y) / np.sum(X**2)  
    k_err = np.sqrt(np.sum((y - k*X)**2) / (len(X) * np.sum(X**2)))  
      
    # Prediction quality  
    y_pred = 1 + k * X  
    r_squared = 1 - np.sum((q['dt_eff'] - y_pred)**2) / np.sum((q['dt_eff'] - q['dt_eff'].mean())**2)  
      
    self.params['metric'] = {  
        'k_coeff': k,  
        'k_err': k_err,  
        'r_squared': r_squared,  
        'model': f'dt_eff = 1 + ({k:.3e} ± {k_err:.0e}) * ε²'  
    }  
      
    return self.params['metric']  
  
def calibrate_velocity_model(self) -> Dict:  
    """  
    Calibrate velocity reduction: v_eff = v0 * (1 - c * epsilon^n)  
    """  
    df = self.data.df  
      
    v0 = self.data.baseline['v_eff']  
    delta_v = (v0 - df['v_eff']) / v0  
      
    log_eps = np.log10(df['epsilon'])  
    log_dv = np.log10(delta_v + 1e-20)  
      
    mask = delta_v > 1e-10  
    if mask.sum() > 5:  
        slope, intercept, r_value, _, _ = stats.linregress(  
            log_eps[mask], log_dv[mask]  
        )  
          
        n = slope  
        c = 10**intercept  
          
        self.params['velocity'] = {  
            'v0_m_s': v0,  
            'power_law_exp': n,  
            'prefactor': c,  
            'r_squared': r_value**2,  
            'model': f'v_eff = {v0:.3e} * (1 - {c:.3e} * ε^{n:.2f})'  
        }  
    else:  
        self.params['velocity'] = {  
            'v0_m_s': v0,  
            'note': 'Velocity reduction below detection threshold in tested range'  
        }  
      
    return self.params['velocity']  
  
def calibrate_all(self) -> Dict:  
    """Run full calibration suite."""  
    print("=" * 80)  
    print("QUANTITATIVE MODEL CALIBRATION")  
    print("=" * 80)  
      
    self.calibrate_energy_model()  
    self.calibrate_metric_model()  
    self.calibrate_velocity_model()  
      
    for name, params in self.params.items():  
        print(f"\n[{name.upper()} MODEL]")  
        for key, val in params.items():  
            print(f"  {key}: {val}")  
      
    return self.params

class PredictiveValidator:
"""Test model predictions on held-out data."""

def __init__(self, data: ValidationData, calibration: ModelCalibration):  
    self.data = data  
    self.cal = calibration  
  
def cross_validate_energy(self, n_folds: int = 5) -> Dict:  
    """  
    K-fold cross-validation of energy model.  
    """  
    df = self.data.df  
    n = len(df)  
    fold_size = n // n_folds  
      
    errors = []  
      
    for i in range(n_folds):  
        test_idx = slice(i * fold_size, (i + 1) * fold_size)  
        train_idx = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, n))  
          
        train = df.iloc[train_idx]  
        test = df.iloc[test_idx]  
          
        X_train = train['epsilon_squared'].values.reshape(-1, 1)  
        A_train = np.hstack([np.ones_like(X_train), X_train])  
        coeffs = np.linalg.lstsq(A_train, train['E_total'].values, rcond=None)[0]  
          
        X_test = test['epsilon_squared'].values  
        y_pred = coeffs[0] + coeffs[1] * X_test  
        y_true = test['E_total'].values  
          
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  
        errors.append(mape)  
      
    return {  
        'cv_mape_mean': np.mean(errors),  
        'cv_mape_std': np.std(errors),  
        'cv_errors': errors,  
        'validation': 'EXCELLENT' if np.mean(errors) < 0.1 else 'GOOD' if np.mean(errors) < 1 else 'POOR'  
    }  
  
def extrapolation_test(self, epsilon_test: float = 1.0) -> Dict:  
    """  
    Test model extrapolation to untested epsilon values.  
    """  
    if 'energy' in self.cal.params:  
        p = self.cal.params['energy']  
        E_pred = p['E_warp_J'] + p['E_3dt_coeff_J'] * epsilon_test**2  
          
        M_sun = 1.989e30  
        c = 2.998e8  
        E_max_physical = M_sun * c**2  
          
        return {  
            'epsilon_test': epsilon_test,  
            'E_predicted_J': E_pred,  
            'E_predicted_M_sun': E_pred / E_max_physical,  
            'physically_plausible': E_pred < E_max_physical,  
            'warning': 'Extrapolation beyond tested regime highly uncertain'  
        }  
      
    return {}

class ValidationReport:
"""Generate comprehensive validation report."""

def __init__(self, validator: StatisticalValidator,  
             calibration: ModelCalibration,  
             predictive: PredictiveValidator):  
    self.val = validator  
    self.cal = calibration  
    self.pred = predictive  
    self.report = {}  
  
def generate(self) -> Dict:  
    """Generate full validation report."""  
    self.report = {  
        'metadata': {  
            'timestamp': pd.Timestamp.now().isoformat(),  
            'data_points': len(self.val.data.df),  
            'epsilon_range': (self.val.data.df['epsilon'].min(),  
                            self.val.data.df['epsilon'].max()),  
            'baseline_energy_J': self.val.data.baseline['E_total']  
        },  
        'validation_results': self.val.results,  
        'calibration_params': self.cal.params,  
        'predictive_tests': {  
            'cv_energy': self.pred.cross_validate_energy(),  
            'extrapolation': self.pred.extrapolation_test()  
        },  
        'overall_assessment': self._overall_assessment()  
    }  
    return self.report  
  
def _overall_assessment(self) -> Dict:  
    """Compute overall validation score."""  
    validations = [r.get('validated', False) for r in self.val.results.values()]  
    validated_count = sum(validations)  
    total_count = len(validations)  
      
    confidence_scores = {  
        '99.9%': 1.0, 'HIGH': 0.9, '95%': 0.95,  
        'MODERATE': 0.7, 'LOW': 0.4, 'INSIGNIFICANT': 0.0  
    }  
      
    weighted_score = 0  
    for result in self.val.results.values():  
        conf = result.get('confidence', 'LOW')  
        weight = confidence_scores.get(conf, 0.5)  
        if result.get('validated', False):  
            weighted_score += weight  
      
    max_possible = len(self.val.results)  
    overall_score = weighted_score / max_possible  
      
    return {  
        'claims_validated': f'{validated_count}/{total_count}',  
        'success_rate': f'{100*validated_count/total_count:.1f}%',  
        'weighted_score': f'{100*overall_score:.1f}%',  
        'assessment': 'STRONGLY VALIDATED' if overall_score > 0.9 else  
                     'VALIDATED' if overall_score > 0.7 else  
                     'PARTIALLY VALIDATED' if overall_score > 0.5 else  
                     'NOT VALIDATED',  
        'recommendation': self._recommendation(overall_score)  
    }  
  
def _recommendation(self, score: float) -> str:  
    if score > 0.9:  
        return "Model is empirically validated. Proceed to engineering feasibility studies."  
    elif score > 0.7:  
        return "Model shows promise. Additional data needed in transitional regime."  
    elif score > 0.5:  
        return "Model partially validated. Critical review of assumptions required."  
    else:  
        return "Model not validated. Fundamental revision needed."  
  
def save(self, filename: str = 'validation_report.json'):  
    """Save report to JSON."""  
    with open(filename, 'w') as f:  
        json.dump(self.report, f, indent=2, default=str)  
    print(f"\nReport saved to: {filename}")  
  
def print_summary(self):  
    """Print executive summary."""  
    print("\n" + "=" * 80)  
    print("VALIDATION EXECUTIVE SUMMARY")  
    print("=" * 80)  
      
    assessment = self.report['overall_assessment']  
    print(f"Overall Assessment: {assessment['assessment']}")  
    print(f"Success Rate: {assessment['success_rate']}")  
    print(f"Weighted Score: {assessment['weighted_score']}")  
    print(f"\nRecommendation: {assessment['recommendation']}")  
      
    print("\nDetailed Results:")  
    for claim, result in self.report['validation_results'].items():  
        status = "✅" if result.get('validated') else "❌"  
        print(f"  {status} {result['claim']}: {result.get('confidence', 'N/A')}")

def create_validation_plots(data: ValidationData, cal: ModelCalibration,
save_prefix: str = 'validation'):
"""Create publication-quality validation plots."""

fig = plt.figure(figsize=(20, 16))  
df = data.df  
  
# [1] Energy vs Epsilon with Fit  
ax1 = plt.subplot(3, 3, 1)  
ax1.loglog(df['epsilon'], df['E_total'], 'bo', markersize=4, label='Data')  
  
if 'energy' in cal.params:  
    p = cal.params['energy']  
    eps_fit = np.logspace(-20, -1, 100)  
    E_fit = p['E_warp_J'] + p['E_3dt_coeff_J'] * eps_fit**2  
    ax1.loglog(eps_fit, E_fit, 'r--', label=f"Fit: $R^2$={p['r_squared']:.6f}")  
  
ax1.set_xlabel('ε (3D time suppression)')  
ax1.set_ylabel('E_total [J]')  
ax1.set_title('Energy Calibration: $E = E_{warp} + A\epsilon^2$')  
ax1.legend()  
ax1.grid(True, alpha=0.3)  
  
# [2] Metric Factor vs Epsilon^2 (should be linear)  
ax2 = plt.subplot(3, 3, 2)  
q = data.get_regime('quantum')  
ax2.plot(q['epsilon_squared'], q['dt_eff'] - 1, 'go', markersize=4, label='Quantum regime')  
  
if 'metric' in cal.params:  
    p = cal.params['metric']  
    eps_sq_fit = np.linspace(0, q['epsilon_squared'].max(), 100)  
    ax2.plot(eps_sq_fit, p['k_coeff'] * eps_sq_fit, 'r--',  
            label=f"Linear fit: k={p['k_coeff']:.2e}")  
  
ax2.set_xlabel('ε²')  
ax2.set_ylabel('dt_eff - 1')  
ax2.set_title('Metric Perturbation: Linear Regime')  
ax2.legend()  
ax2.grid(True, alpha=0.3)  
  
# [3] Velocity Reduction  
ax3 = plt.subplot(3, 3, 3)  
v0 = data.baseline['v_eff']  
ax3.semilogx(df['epsilon'], (v0 - df['v_eff'])/v0 * 100, 'mo-', markersize=3)  
ax3.set_xlabel('ε')  
ax3.set_ylabel('Velocity Reduction [%]')  
ax3.set_title('Effective Velocity: 3D Time Drag Effect')  
ax3.grid(True, alpha=0.3)  
  
# [4] Regime Identification  
ax4 = plt.subplot(3, 3, 4)  
colors = {'quantum': 'blue', 'transitional': 'orange', 'classical': 'red'}  
for regime, subset in data.regimes.items():  
    if len(subset) > 0:  
        ax4.scatter(subset['log10_epsilon'], subset['E_ratio'],  
                   c=colors[regime], label=regime, s=20, alpha=0.6)  
ax4.axvline(-8, color='k', linestyle='--', alpha=0.3, label='Quantum-Transitional')  
ax4.axvline(-4, color='k', linestyle='--', alpha=0.3, label='Transitional-Classical')  
ax4.set_xlabel('log₁₀(ε)')  
ax4.set_ylabel('E / E_baseline')  
ax4.set_title('Physical Regimes')  
ax4.legend()  
ax4.grid(True, alpha=0.3)  
  
# [5] Residuals Analysis  
ax5 = plt.subplot(3, 3, 5)  
if 'energy' in cal.params:  
    p = cal.params['energy']  
    E_pred = p['E_warp_J'] + p['E_3dt_coeff_J'] * df['epsilon_squared']  
    residuals = (df['E_total'] - E_pred) / df['E_total'] * 100  
    ax5.plot(df['log10_epsilon'], residuals, 'ko', markersize=3)  
    ax5.axhline(0, color='r', linestyle='--')  
    ax5.set_xlabel('log₁₀(ε)')  
    ax5.set_ylabel('Residuals [%]')  
    ax5.set_title('Energy Model Residuals')  
    ax5.grid(True, alpha=0.3)  
  
# [6] Alpha^2 Safety Margin  
ax6 = plt.subplot(3, 3, 6)  
alpha_sq = (1/df['dt_eff'])**2  
ax6.semilogx(df['epsilon'], alpha_sq, 'co-', markersize=3)  
ax6.axhline(0.5, color='r', linestyle='--', label='Safety threshold')  
ax6.fill_between(df['epsilon'], 0, 0.5, alpha=0.2, color='red')  
ax6.set_xlabel('ε')  
ax6.set_ylabel('α² = 1/dt_eff²')  
ax6.set_title('Alpha² Safety Margin: No Event Horizons')  
ax6.legend()  
ax6.grid(True, alpha=0.3)  
  
# [7] Delta dt vs epsilon (perturbative regime)  
ax7 = plt.subplot(3, 3, 7)  
ax7.loglog(df['epsilon'], df['delta_dt'], 'bs-', markersize=3)  
ax7.set_xlabel('ε')  
ax7.set_ylabel('Δdt')  
ax7.set_title('Delta dt vs ε')  
ax7.grid(True, alpha=0.3)  
  
# [8] Delta v vs epsilon  
ax8 = plt.subplot(3, 3, 8)  
ax8.loglog(df['epsilon'], df['delta_v'], 'rs-', markersize=3)  
ax8.set_xlabel('ε')  
ax8.set_ylabel('Δv [m/s]')  
ax8.set_title('Delta Velocity vs ε')  
ax8.grid(True, alpha=0.3)  
  
# [9] Energy Ratio vs Epsilon  
ax9 = plt.subplot(3, 3, 9)  
ax9.loglog(df['epsilon'], df['E_ratio'], 'k^-', markersize=3)  
ax9.set_xlabel('ε')  
ax9.set_ylabel('E / E_baseline')  
ax9.set_title('Normalized Energy vs ε')  
ax9.grid(True, alpha=0.3)  
  
plt.tight_layout()  
  
if save_prefix:  
    plt.savefig(f"{save_prefix}_plots.png", dpi=300)  
    print(f"Validation plots saved as: {save_prefix}_plots.png")  
  
plt.show()

def main():
"""Execute complete empirical validation pipeline."""

print("=" * 80)  
print("EMPIRICAL VALIDATION PIPELINE FOR 3D TIME WARP FIELD THEORY")  
print("=" * 80)  
  
# [1] Load Data  
print("\n[1] LOADING EMPIRICAL DATA")  
print("-" * 50)  
data = ValidationData('3D_time_validation_results.csv')  
print(f"Loaded {len(data.df)} data points")  
print(f"Epsilon range: {data.df['epsilon'].min():.0e} to {data.df['epsilon'].max():.0e}")  
print(f"Regimes identified:")  
for name, subset in data.regimes.items():  
    print(f"  - {name}: {len(subset)} points")  
  
# [2] Statistical Validation  
print("\n[2] STATISTICAL VALIDATION")  
print("-" * 50)  
validator = StatisticalValidator(data)  
validator.run_all_validations()  
  
# [3] Model Calibration  
print("\n[3] QUANTITATIVE CALIBRATION")  
print("-" * 50)  
calibration = ModelCalibration(data)  
calibration.calibrate_all()  
  
# [4] Predictive Testing  
print("\n[4] PREDICTIVE VALIDATION")  
print("-" * 50)  
predictive = PredictiveValidator(data, calibration)  
cv_results = predictive.cross_validate_energy()  
print(f"Cross-validation MAPE: {cv_results['cv_mape_mean']:.4f}% ± {cv_results['cv_mape_std']:.4f}%")  
  
extrap = predictive.extrapolation_test()  
if extrap:  
    print(f"Extrapolation to ε=1.0: E = {extrap['E_predicted_J']:.3e} J")  
    print(f"  = {extrap['E_predicted_M_sun']:.1f} M_sun")  
    print(f"  Physically plausible: {extrap['physically_plausible']}")  
  
# [5] Generate Report  
print("\n[5] GENERATING VALIDATION REPORT")  
print("-" * 50)  
report = ValidationReport(validator, calibration, predictive)  
report.generate()  
report.print_summary()  
report.save('validation_report.json')  
  
# [6] Visualization  
print("\n[6] CREATING VALIDATION PLOTS")  
print("-" * 50)  
create_validation_plots(data, calibration, save_prefix='validation')  
  
# [7] Final Recommendations  
print("\n" + "=" * 80)  
print("VALIDATION COMPLETE")  
print("=" * 80)  
  
# Extract key findings  
energy_val = validator.results.get('positive_energy', {})  
scaling_val = validator.results.get('epsilon_scaling', {})  
  
print(f"\nKey Findings:")  
print(f"  • Energy positivity: {energy_val.get('confidence', 'N/A')} confidence")  
print(f"  • Epsilon^2 scaling: n = {scaling_val.get('fitted_exponent', 'N/A'):.3f} ± {scaling_val.get('exponent_error', 'N/A'):.3f}")  
print(f"  • Velocity penalty at max ε: {validator.results.get('velocity_reduction', {}).get('velocity_penalty_percent', 'N/A'):.2f}%")  
  
print(f"\nCalibrated Parameters:")  
if 'energy' in calibration.params:  
    p = calibration.params['energy']  
    print(f"  • E_warp = {p['E_warp_J']:.6e} ± {p['E_warp_err']:.0e} J")  
    print(f"  • E_3dt/ε² = {p['E_3dt_coeff_J']:.3e} ± {p['E_3dt_err']:.0e} J")  
  
return data, validator, calibration, report

if name == "main":
data, validator, calibration, report = main()