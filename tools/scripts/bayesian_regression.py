#!/usr/bin/env python3
"""
Bayesian Regression Analysis for Race Performance Data
Uses Bayesian inference for performance modeling with uncertainty quantification.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import PyMC3 for advanced Bayesian modeling
try:
    import pymc3 as pm
    import theano.tensor as tt
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False
    print("PyMC3 not available. Using sklearn BayesianRidge as fallback.")

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/bayesian_regression']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare for Bayesian analysis."""
    # Load the scraped race data
    df = pd.read_csv('data/raw/race_results.csv')
    
    # Convert time strings to seconds
    def time_to_seconds(time_str):
        if pd.isna(time_str) or time_str == '':
            return np.nan
        try:
            parts = str(time_str).split(':')
            if len(parts) == 2:  # MM:SS format
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS format
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return np.nan
        except:
            return np.nan
    
    # Process time columns
    time_columns = ['gun_time', 'chip_time', '10km']
    for col in time_columns:
        if col in df.columns:
            df[f'{col}_seconds'] = df[col].apply(time_to_seconds)
    
    # Target variable
    df['finish_time'] = df['chip_time_seconds'].fillna(df['gun_time_seconds'])
    df['log_finish_time'] = np.log(df['finish_time'])  # Log transform for normality
    
    # Extract age and demographics
    def extract_age(category):
        if pd.isna(category):
            return np.nan
        try:
            age_part = ''.join(filter(str.isdigit, str(category)))
            return int(age_part) if age_part else np.nan
        except:
            return np.nan
    
    df['age'] = df['category'].apply(extract_age)
    df['age_centered'] = df['age'] - df['age'].mean()  # Center age for better interpretation
    df['age_squared'] = df['age_centered'] ** 2
    
    # Gender
    df['gender'] = df['gender'].str.upper()
    df['is_male'] = (df['gender'] == 'MALE').astype(int)
    
    # Club features
    df['has_club'] = (~df['club'].isna() & (df['club'] != 'None') & (df['club'] != '')).astype(int)
    
    # Calculate club effects for hierarchical modeling
    club_stats = df.groupby('club').agg({
        'finish_time': ['count', 'mean', 'std'],
        'age': 'mean'
    }).round(2)
    
    club_stats.columns = ['club_size', 'club_avg_time', 'club_time_std', 'club_avg_age']
    club_stats = club_stats.reset_index()
    
    # Only keep clubs with sufficient data for meaningful priors
    club_stats = club_stats[club_stats['club_size'] >= 3]
    
    df = df.merge(club_stats, on='club', how='left')
    df['club_size'] = df['club_size'].fillna(1)
    df['club_avg_time'] = df['club_avg_time'].fillna(df['finish_time'].median())
    
    # Large club indicator
    df['large_club'] = (df['club_size'] >= 10).astype(int)
    
    # Pacing features if available
    if '10km_seconds' in df.columns:
        df['has_10km_split'] = (~df['10km_seconds'].isna()).astype(int)
        df['pace_10km'] = df['10km_seconds'] / 10000  # seconds per meter
        
        # Only calculate pace ratio for those with split times
        pace_mask = df['10km_seconds'].notna()
        df.loc[pace_mask, 'pace_overall'] = df.loc[pace_mask, 'finish_time'] / 21097
        df.loc[pace_mask, 'pace_ratio'] = (
            (df.loc[pace_mask, 'finish_time'] - df.loc[pace_mask, '10km_seconds']) / 11097
        ) / df.loc[pace_mask, 'pace_10km']
        
        # Fill missing pace data
        df['pace_ratio'] = df['pace_ratio'].fillna(1.1)  # Typical positive split
    else:
        df['has_10km_split'] = 0
        df['pace_ratio'] = 1.1
    
    return df

def prepare_bayesian_features(df):
    """Prepare features specifically for Bayesian modeling."""
    
    # Core features for Bayesian regression
    features = [
        'age_centered', 'age_squared', 'is_male', 'has_club', 'large_club',
        'has_10km_split', 'pace_ratio'
    ]
    
    # Add 10km split if available
    if 'pace_10km' in df.columns:
        features.append('pace_10km')
    
    # Create feature matrix
    X = df[features].copy()
    
    # Target variables (both regular and log-transformed)
    y = df['finish_time']
    y_log = df['log_finish_time']
    
    # Remove missing values
    complete_mask = y.notna() & df['age'].notna()
    X_clean = X[complete_mask]
    y_clean = y[complete_mask]
    y_log_clean = y_log[complete_mask]
    
    # Fill remaining missing values
    X_clean = X_clean.fillna(X_clean.median())
    
    return X_clean, y_clean, y_log_clean, complete_mask

def fit_bayesian_ridge_models(X, y, y_log):
    """Fit Bayesian Ridge regression models using sklearn."""
    
    print("Fitting Bayesian Ridge regression models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    y_log_train = np.log(y_train)
    y_log_test = np.log(y_test)
    
    models = {}
    results = {}
    
    # Model 1: Standard Bayesian Ridge
    br_standard = BayesianRidge(
        n_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        compute_score=True
    )
    
    br_standard.fit(X_train, y_train)
    models['standard'] = br_standard
    
    # Predictions with uncertainty
    y_pred_train, y_std_train = br_standard.predict(X_train, return_std=True)
    y_pred_test, y_std_test = br_standard.predict(X_test, return_std=True)
    
    results['standard'] = {
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'alpha_': float(br_standard.alpha_),
        'lambda_': float(br_standard.lambda_),
        'coefficients': dict(zip(X.columns, br_standard.coef_)),
        'intercept': float(br_standard.intercept_),
        'log_marginal_likelihood': float(br_standard.scores_[-1]) if len(br_standard.scores_) > 0 else None,
        'prediction_std_mean': float(y_std_test.mean()),
        'prediction_std_median': float(np.median(y_std_test))
    }
    
    # Model 2: Log-transformed target
    br_log = BayesianRidge(
        n_iter=300,
        tol=1e-3,
        compute_score=True
    )
    
    br_log.fit(X_train, y_log_train)
    models['log_transformed'] = br_log
    
    # Predictions (transform back to original scale)
    y_log_pred_train, y_log_std_train = br_log.predict(X_train, return_std=True)
    y_log_pred_test, y_log_std_test = br_log.predict(X_test, return_std=True)
    
    # Transform back to original scale
    y_pred_train_log = np.exp(y_log_pred_train)
    y_pred_test_log = np.exp(y_log_pred_test)
    
    results['log_transformed'] = {
        'train_mae': mean_absolute_error(y_train, y_pred_train_log),
        'test_mae': mean_absolute_error(y_test, y_pred_test_log),
        'train_r2': r2_score(y_train, y_pred_train_log),
        'test_r2': r2_score(y_test, y_pred_test_log),
        'alpha_': float(br_log.alpha_),
        'lambda_': float(br_log.lambda_),
        'coefficients': dict(zip(X.columns, br_log.coef_)),
        'intercept': float(br_log.intercept_),
        'log_marginal_likelihood': float(br_log.scores_[-1]) if len(br_log.scores_) > 0 else None,
        'log_prediction_std_mean': float(y_log_std_test.mean())
    }
    
    # Store predictions for analysis
    results['predictions'] = {
        'y_test': y_test.tolist(),
        'y_pred_standard': y_pred_test.tolist(),
        'y_std_standard': y_std_test.tolist(),
        'y_pred_log': y_pred_test_log.tolist(),
        'y_std_log': y_log_std_test.tolist()
    }
    
    return models, results, (X_train, X_test, y_train, y_test)

def fit_pymc3_model(X, y):
    """Fit hierarchical Bayesian model using PyMC3 (if available)."""
    
    if not PYMC3_AVAILABLE:
        return None, {'error': 'PyMC3 not available'}
    
    print("Fitting hierarchical Bayesian model with PyMC3...")
    
    try:
        # Standardize features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Take a subset for faster computation
        n_samples = min(1000, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X_scaled[indices]
        y_subset = y.iloc[indices].values
        
        with pm.Model() as hierarchical_model:
            # Priors for regression coefficients
            beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])
            alpha = pm.Normal('alpha', mu=np.log(y_subset.mean()), sigma=1)
            
            # Prior for noise
            sigma = pm.HalfNormal('sigma', sigma=y_subset.std())
            
            # Linear model
            mu = alpha + pm.math.dot(X_subset, beta)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y_subset)
            
            # Sample
            trace = pm.sample(1000, tune=500, cores=1, progressbar=True)
        
        # Extract results
        pymc3_results = {
            'summary': pm.summary(trace).to_dict(),
            'coefficients': {f'beta_{i}': float(trace['beta'][:, i].mean()) 
                           for i in range(X.shape[1])},
            'coefficient_std': {f'beta_{i}': float(trace['beta'][:, i].std()) 
                              for i in range(X.shape[1])},
            'intercept_mean': float(trace['alpha'].mean()),
            'intercept_std': float(trace['alpha'].std()),
            'sigma_mean': float(trace['sigma'].mean()),
            'sigma_std': float(trace['sigma'].std()),
            'effective_sample_size': pm.ess(trace).to_dict(),
            'r_hat': pm.rhat(trace).to_dict()
        }
        
        return hierarchical_model, pymc3_results, trace
    
    except Exception as e:
        print(f"PyMC3 modeling failed: {e}")
        return None, {'error': str(e)}, None

def analyze_bayesian_results(models, results, X, y):
    """Analyze Bayesian regression results and extract insights."""
    
    analysis = {}
    
    print("Analyzing Bayesian results...")
    
    # 1. Model comparison
    model_comparison = {}
    for model_name, model_results in results.items():
        if model_name != 'predictions' and 'error' not in model_results:
            model_comparison[model_name] = {
                'test_mae': model_results.get('test_mae', 0) / 60,  # Convert to minutes
                'test_r2': model_results.get('test_r2', 0),
                'log_marginal_likelihood': model_results.get('log_marginal_likelihood', 0)
            }
    
    analysis['model_comparison'] = model_comparison
    
    # 2. Coefficient analysis with uncertainty
    if 'standard' in results:
        coeffs = results['standard']['coefficients']
        
        # Calculate coefficient interpretations
        coefficient_insights = {}
        for feature, coef in coeffs.items():
            coefficient_insights[feature] = {
                'coefficient': coef,
                'interpretation': interpret_coefficient(feature, coef),
                'magnitude': abs(coef),
                'direction': 'positive' if coef > 0 else 'negative'
            }
        
        analysis['coefficient_analysis'] = coefficient_insights
    
    # 3. Uncertainty quantification
    if 'predictions' in results:
        preds = results['predictions']
        
        # Prediction intervals
        y_pred = np.array(preds['y_pred_standard'])
        y_std = np.array(preds['y_std_standard'])
        y_test = np.array(preds['y_test'])
        
        # Calculate prediction intervals
        prediction_intervals = {
            '68_percent': {
                'lower': (y_pred - y_std).tolist(),
                'upper': (y_pred + y_std).tolist()
            },
            '95_percent': {
                'lower': (y_pred - 1.96 * y_std).tolist(),
                'upper': (y_pred + 1.96 * y_std).tolist()
            }
        }
        
        # Coverage analysis
        in_68 = np.sum((y_test >= y_pred - y_std) & (y_test <= y_pred + y_std)) / len(y_test)
        in_95 = np.sum((y_test >= y_pred - 1.96*y_std) & (y_test <= y_pred + 1.96*y_std)) / len(y_test)
        
        uncertainty_analysis = {
            'prediction_intervals': prediction_intervals,
            'coverage_68_percent': float(in_68),
            'coverage_95_percent': float(in_95),
            'mean_prediction_std_minutes': float(y_std.mean() / 60),
            'median_prediction_std_minutes': float(np.median(y_std) / 60)
        }
        
        analysis['uncertainty_analysis'] = uncertainty_analysis
    
    # 4. Performance predictions with uncertainty
    scenarios = generate_prediction_scenarios(X)
    
    scenario_predictions = {}
    if 'standard' in models:
        model = models['standard']
        
        for name, scenario in scenarios.items():
            try:
                pred_mean, pred_std = model.predict(scenario.reshape(1, -1), return_std=True)
                
                scenario_predictions[name] = {
                    'mean_seconds': float(pred_mean[0]),
                    'mean_minutes': float(pred_mean[0] / 60),
                    'std_seconds': float(pred_std[0]),
                    'std_minutes': float(pred_std[0] / 60),
                    'confidence_interval_95': {
                        'lower_minutes': float((pred_mean[0] - 1.96*pred_std[0]) / 60),
                        'upper_minutes': float((pred_mean[0] + 1.96*pred_std[0]) / 60)
                    }
                }
            except Exception as e:
                scenario_predictions[name] = {'error': str(e)}
    
    analysis['scenario_predictions'] = scenario_predictions
    
    # 5. Prior sensitivity analysis
    if 'standard' in results and 'log_transformed' in results:
        sensitivity_analysis = {
            'regularization_effect': {
                'alpha_standard': results['standard']['alpha_'],
                'alpha_log': results['log_transformed']['alpha_'],
                'lambda_standard': results['standard']['lambda_'],
                'lambda_log': results['log_transformed']['lambda_']
            },
            'model_evidence_comparison': {
                'standard_likelihood': results['standard'].get('log_marginal_likelihood', 0),
                'log_likelihood': results['log_transformed'].get('log_marginal_likelihood', 0)
            }
        }
        
        analysis['sensitivity_analysis'] = sensitivity_analysis
    
    return analysis

def interpret_coefficient(feature, coefficient):
    """Provide interpretation for regression coefficients."""
    
    interpretations = {
        'age_centered': f"Each year older/younger than average changes finish time by {coefficient:.1f} seconds",
        'age_squared': f"Age effect {'accelerates' if coefficient > 0 else 'decelerates'} with age",
        'is_male': f"Males finish {abs(coefficient):.1f} seconds {'faster' if coefficient < 0 else 'slower'} than females",
        'has_club': f"Club membership changes finish time by {coefficient:.1f} seconds",
        'large_club': f"Large club membership effect: {coefficient:.1f} seconds",
        'pace_ratio': f"Each unit increase in pace ratio changes finish time by {coefficient:.1f} seconds",
        'pace_10km': f"Each second/meter increase in 10km pace changes finish time by {coefficient:.1f} seconds"
    }
    
    return interpretations.get(feature, f"Effect size: {coefficient:.3f}")

def generate_prediction_scenarios(X):
    """Generate prediction scenarios for different runner types."""
    
    feature_means = X.mean()
    scenarios = {}
    
    # Young competitive male
    scenario1 = feature_means.copy()
    scenario1['age_centered'] = 25 - 45  # Assuming 45 is mean age
    scenario1['age_squared'] = scenario1['age_centered'] ** 2
    scenario1['is_male'] = 1
    scenario1['has_club'] = 1
    scenario1['large_club'] = 1
    if 'pace_ratio' in scenario1.index:
        scenario1['pace_ratio'] = 1.05  # Slight positive split
    scenarios['Young Competitive Male'] = scenario1.values
    
    # Middle-aged recreational female
    scenario2 = feature_means.copy()
    scenario2['age_centered'] = 45 - 45  # At mean age
    scenario2['age_squared'] = 0
    scenario2['is_male'] = 0
    scenario2['has_club'] = 0
    scenario2['large_club'] = 0
    if 'pace_ratio' in scenario2.index:
        scenario2['pace_ratio'] = 1.15  # Moderate positive split
    scenarios['Middle-aged Recreational Female'] = scenario2.values
    
    # Veteran club runner
    scenario3 = feature_means.copy()
    scenario3['age_centered'] = 60 - 45  # 15 years above mean
    scenario3['age_squared'] = scenario3['age_centered'] ** 2
    scenario3['is_male'] = 1
    scenario3['has_club'] = 1
    scenario3['large_club'] = 1
    if 'pace_ratio' in scenario3.index:
        scenario3['pace_ratio'] = 1.10  # Moderate positive split
    scenarios['Veteran Club Runner'] = scenario3.values
    
    return scenarios

def save_results(models, results, analysis, pymc3_results, X, y):
    """Save all Bayesian analysis results."""
    
    # Save processed data
    feature_target_df = X.copy()
    feature_target_df['finish_time'] = y
    feature_target_df.to_csv('data/processed/bayesian_regression_data.csv', index=False)
    
    # Combine all results
    full_results = {
        'model_results': results,
        'analysis': analysis,
        'pymc3_results': pymc3_results if pymc3_results and 'error' not in pymc3_results else None,
        'data_summary': {
            'total_samples': len(X),
            'num_features': len(X.columns),
            'target_mean_minutes': float(y.mean() / 60),
            'target_std_minutes': float(y.std() / 60),
            'feature_list': X.columns.tolist()
        }
    }
    
    # Save results to JSON
    with open('data/processed/bayesian_regression_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save model objects
    models_to_save = {k: v for k, v in models.items() if v is not None}
    with open('data/processed/bayesian_regression_models.pkl', 'wb') as f:
        pickle.dump(models_to_save, f)
    
    print("Results saved to data/processed/")
    
    # Print summary
    print(f"\nBayesian Regression Analysis Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Features used: {len(X.columns)}")
    
    if 'standard' in results:
        print(f"Standard model R²: {results['standard']['test_r2']:.3f}")
        print(f"Standard model MAE: {results['standard']['test_mae']/60:.1f} minutes")
        print(f"Mean prediction uncertainty: ±{results['standard']['prediction_std_mean']/60:.1f} minutes")
    
    if 'model_comparison' in analysis:
        best_model = max(analysis['model_comparison'].items(), key=lambda x: x[1]['test_r2'])
        print(f"Best model: {best_model[0]} (R² = {best_model[1]['test_r2']:.3f})")

def main():
    """Main execution function."""
    print("Starting Bayesian Regression Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Prepare features
    print("Preparing features for Bayesian modeling...")
    X, y, y_log, complete_cases = prepare_bayesian_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Fit Bayesian Ridge models
    models, results, splits = fit_bayesian_ridge_models(X, y, y_log)
    
    # Try PyMC3 hierarchical model
    pymc3_model, pymc3_results, pymc3_trace = fit_pymc3_model(X, y)
    
    # Analyze results
    print("Analyzing Bayesian results...")
    analysis = analyze_bayesian_results(models, results, X, y)
    
    # Save results
    save_results(models, results, analysis, pymc3_results, X, y)
    
    print("Bayesian regression analysis complete!")

if __name__ == "__main__":
    main()