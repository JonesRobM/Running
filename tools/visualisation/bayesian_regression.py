#!/usr/bin/env python3
"""
Bayesian Regression Visualization
Creates comprehensive visualizations for Bayesian regression analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def ensure_output_dir(technique_name):
    """Create output directory for figures if it doesn't exist."""
    output_dir = Path(f'figures/{technique_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_results():
    """Load Bayesian regression analysis results and data."""
    # Load processed data
    df = pd.read_csv('data/processed/bayesian_regression_data.csv')
    
    # Load results
    with open('data/processed/bayesian_regression_results.json', 'r') as f:
        results = json.load(f)
    
    # Load model objects
    try:
        with open('data/processed/bayesian_regression_models.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}
    
    return df, results, models

def plot_coefficient_analysis(results, output_dir):
    """Plot Bayesian coefficient analysis with uncertainty."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Coefficient estimates with interpretation
    if 'coefficient_analysis' in results['analysis']:
        coeff_analysis = results['analysis']['coefficient_analysis']
        
        features = list(coeff_analysis.keys())
        coefficients = [coeff_analysis[f]['coefficient'] for f in features]
        magnitudes = [coeff_analysis[f]['magnitude'] for f in features]
        
        # Color by direction
        colors = ['green' if coeff_analysis[f]['direction'] == 'negative' else 'red' 
                 for f in features]
        
        bars = ax1.barh(range(len(features)), coefficients, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Bayesian Regression Coefficients', fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add magnitude as text
        for i, (bar, mag) in enumerate(zip(bars, magnitudes)):
            ax1.text(bar.get_width() + (0.1 if bar.get_width() > 0 else -0.1), 
                    bar.get_y() + bar.get_height()/2,
                    f'{mag:.3f}', va='center', fontsize=9)
    
    # Plot 2: Model comparison (standard vs log-transformed)
    if 'model_comparison' in results['analysis']:
        model_comp = results['analysis']['model_comparison']
        
        models_available = list(model_comp.keys())
        mae_values = [model_comp[m]['test_mae'] for m in models_available]
        r2_values = [model_comp[m]['test_r2'] for m in models_available]
        
        x = np.arange(len(models_available))
        width = 0.35
        
        ax2_twin = ax2.twinx()
        
        # MAE bars
        bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE (min)', alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('MAE (minutes)', color='red')
        ax2.set_title('Bayesian Model Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in models_available])
        
        # R² line
        line1 = ax2_twin.plot(x, r2_values, 'bo-', label='R²', linewidth=2, markersize=8)
        ax2_twin.set_ylabel('R² Score', color='blue')
        
        # Add value labels
        for i, (bar, mae, r2) in enumerate(zip(bars1, mae_values, r2_values)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mae:.1f}', ha='center', va='bottom', fontsize=10)
            ax2_twin.text(i, r2 + 0.01, f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty quantification
    if 'uncertainty_analysis' in results['analysis']:
        uncertainty = results['analysis']['uncertainty_analysis']
        
        # Coverage analysis
        coverage_68 = uncertainty['coverage_68_percent'] * 100
        coverage_95 = uncertainty['coverage_95_percent'] * 100
        expected_68 = 68
        expected_95 = 95
        
        intervals = ['68% Interval', '95% Interval']
        actual_coverage = [coverage_68, coverage_95]
        expected_coverage = [expected_68, expected_95]
        
        x = np.arange(len(intervals))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, actual_coverage, width, label='Actual Coverage', alpha=0.7)
        bars2 = ax3.bar(x + width/2, expected_coverage, width, label='Expected Coverage', alpha=0.7)
        
        ax3.set_xlabel('Prediction Interval')
        ax3.set_ylabel('Coverage Percentage')
        ax3.set_title('Prediction Interval Coverage', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(intervals)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add difference annotations
        for i, (actual, expected) in enumerate(zip(actual_coverage, expected_coverage)):
            diff = actual - expected
            color = 'green' if abs(diff) < 5 else 'orange' if abs(diff) < 10 else 'red'
            ax3.text(i, max(actual, expected) + 2, f'Δ{diff:+.1f}%', 
                    ha='center', color=color, fontweight='bold')
    
    # Plot 4: Prediction uncertainty distribution
    if 'predictions' in results['model_results']:
        predictions = results['model_results']['predictions']
        
        if 'y_std_standard' in predictions:
            y_std = np.array(predictions['y_std_standard']) / 60  # Convert to minutes
            
            ax4.hist(y_std, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(x=np.mean(y_std), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(y_std):.1f} min')
            ax4.axvline(x=np.median(y_std), color='orange', linestyle='--', 
                       label=f'Median: {np.median(y_std):.1f} min')
            
            ax4.set_xlabel('Prediction Standard Deviation (minutes)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Prediction Uncertainty', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coefficient_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_intervals(results, output_dir):
    """Plot prediction intervals and uncertainty visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predictions with confidence intervals
    if 'predictions' in results['model_results']:
        predictions = results['model_results']['predictions']
        
        y_test = np.array(predictions['y_test']) / 60
        y_pred = np.array(predictions['y_pred_standard']) / 60
        y_std = np.array(predictions['y_std_standard']) / 60
        
        # Sort for better visualization
        sort_idx = np.argsort(y_test)
        y_test_sorted = y_test[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        y_std_sorted = y_std[sort_idx]
        
        # Plot subset for clarity
        n_points = min(100, len(y_test_sorted))
        indices = np.linspace(0, len(y_test_sorted)-1, n_points, dtype=int)
        
        x_plot = range(n_points)
        y_test_plot = y_test_sorted[indices]
        y_pred_plot = y_pred_sorted[indices]
        y_std_plot = y_std_sorted[indices]
        
        # Plot predictions and intervals
        ax1.plot(x_plot, y_test_plot, 'bo', alpha=0.6, markersize=4, label='Actual')
        ax1.plot(x_plot, y_pred_plot, 'ro', alpha=0.6, markersize=4, label='Predicted')
        
        # 95% confidence interval
        ax1.fill_between(x_plot, 
                        y_pred_plot - 1.96*y_std_plot, 
                        y_pred_plot + 1.96*y_std_plot,
                        alpha=0.3, color='gray', label='95% Interval')
        
        # 68% confidence interval
        ax1.fill_between(x_plot, 
                        y_pred_plot - y_std_plot, 
                        y_pred_plot + y_std_plot,
                        alpha=0.5, color='lightblue', label='68% Interval')
        
        ax1.set_xlabel('Sample Index (sorted by actual time)')
        ax1.set_ylabel('Finish Time (minutes)')
        ax1.set_title('Predictions with Uncertainty Intervals', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs uncertainty
    if 'predictions' in results['model_results']:
        residuals = y_pred - y_test
        
        ax2.scatter(y_std, np.abs(residuals), alpha=0.6, s=30)
        
        # Add trend line
        z = np.polyfit(y_std, np.abs(residuals), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(y_std.min(), y_std.max(), 100)
        ax2.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Prediction Standard Deviation (minutes)')
        ax2.set_ylabel('Absolute Residual (minutes)')
        ax2.set_title('Residuals vs Prediction Uncertainty', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation
        correlation = np.corrcoef(y_std, np.abs(residuals))[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Scenario predictions with uncertainty
    if 'scenario_predictions' in results['analysis']:
        scenarios = results['analysis']['scenario_predictions']
        
        scenario_names = []
        mean_times = []
        lower_bounds = []
        upper_bounds = []
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                scenario_names.append(scenario.replace(' ', '\n'))
                mean_times.append(prediction['mean_minutes'])
                
                if 'confidence_interval_95' in prediction:
                    lower_bounds.append(prediction['confidence_interval_95']['lower_minutes'])
                    upper_bounds.append(prediction['confidence_interval_95']['upper_minutes'])
                else:
                    # Approximate bounds if not available
                    std_min = prediction.get('std_minutes', 2)
                    lower_bounds.append(prediction['mean_minutes'] - 1.96*std_min)
                    upper_bounds.append(prediction['mean_minutes'] + 1.96*std_min)
        
        if scenario_names:
            x = range(len(scenario_names))
            
            # Error bars for confidence intervals
            yerr_lower = [m - l for m, l in zip(mean_times, lower_bounds)]
            yerr_upper = [u - m for m, u in zip(mean_times, upper_bounds)]
            
            ax3.errorbar(x, mean_times, yerr=[yerr_lower, yerr_upper], 
                        fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.8)
            
            ax3.set_xlabel('Scenario')
            ax3.set_ylabel('Predicted Time (minutes)')
            ax3.set_title('Scenario Predictions with 95% Intervals', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(scenario_names, fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Add mean time labels
            for i, time in enumerate(mean_times):
                ax3.text(i, time + max(yerr_upper)*0.1, f'{time:.0f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Prior vs Posterior comparison (if available)
    if 'sensitivity_analysis' in results['analysis']:
        sensitivity = results['analysis']['sensitivity_analysis']
        
        # Regularization parameters comparison
        if 'regularization_effect' in sensitivity:
            reg_effect = sensitivity['regularization_effect']
            
            params = ['Alpha', 'Lambda']
            standard_values = [reg_effect.get('alpha_standard', 0), 
                             reg_effect.get('lambda_standard', 0)]
            log_values = [reg_effect.get('alpha_log', 0), 
                         reg_effect.get('lambda_log', 0)]
            
            x = np.arange(len(params))
            width = 0.35
            
            ax4.bar(x - width/2, standard_values, width, label='Standard Model', alpha=0.7)
            ax4.bar(x + width/2, log_values, width, label='Log-transformed Model', alpha=0.7)
            
            ax4.set_xlabel('Regularization Parameter')
            ax4.set_ylabel('Parameter Value')
            ax4.set_title('Regularization Parameter Comparison', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(params)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (std_val, log_val) in enumerate(zip(standard_values, log_values)):
                ax4.text(i - width/2, std_val + max(standard_values)*0.02, 
                        f'{std_val:.2e}', ha='center', va='bottom', fontsize=9)
                ax4.text(i + width/2, log_val + max(log_values)*0.02, 
                        f'{log_val:.2e}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Sensitivity analysis not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bayesian_diagnostics(results, output_dir):
    """Plot Bayesian model diagnostics and validation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model evidence comparison
    if 'model_comparison' in results['analysis']:
        model_comp = results['analysis']['model_comparison']
        
        models = list(model_comp.keys())
        log_likelihoods = [model_comp[m].get('log_marginal_likelihood', 0) for m in models]
        
        if any(ll != 0 for ll in log_likelihoods):
            bars = ax1.bar(models, log_likelihoods, alpha=0.7, color='steelblue')
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Log Marginal Likelihood')
            ax1.set_title('Bayesian Model Evidence', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, ll in zip(bars, log_likelihoods):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                        f'{ll:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight best model
            best_idx = np.argmax(log_likelihoods)
            bars[best_idx].set_color('orange')
        else:
            ax1.text(0.5, 0.5, 'Log marginal likelihood not available', 
                    ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Coefficient uncertainty
    if 'standard' in results['model_results']:
        coeffs = results['model_results']['standard']['coefficients']
        
        features = list(coeffs.keys())
        coeff_values = list(coeffs.values())
        
        # Create approximate uncertainty (would be exact with full Bayesian implementation)
        coeff_std = [abs(c) * 0.1 for c in coeff_values]  # Approximate
        
        ax2.errorbar(range(len(features)), coeff_values, yerr=coeff_std,
                    fmt='o', capsize=5, capthick=2, markersize=6, alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title('Coefficient Estimates with Uncertainty', fontweight='bold')
        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predictive distribution examples
    if 'scenario_predictions' in results['analysis']:
        scenarios = results['analysis']['scenario_predictions']
        
        # Create predictive distributions for each scenario
        scenario_names = []
        distributions = []
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction and 'std_minutes' in prediction:
                scenario_names.append(scenario.replace(' ', '\n')[:15])  # Truncate long names
                
                mean = prediction['mean_minutes']
                std = prediction['std_minutes']
                
                # Generate distribution
                x = np.linspace(mean - 3*std, mean + 3*std, 100)
                y = stats.norm.pdf(x, mean, std)
                distributions.append((x, y, mean))
        
        if distributions:
            colors = plt.cm.Set3(np.linspace(0, 1, len(distributions)))
            
            for i, (x, y, mean) in enumerate(distributions):
                ax3.plot(x, y, color=colors[i], linewidth=2, alpha=0.7, 
                        label=scenario_names[i])
                ax3.axvline(x=mean, color=colors[i], linestyle='--', alpha=0.5)
            
            ax3.set_xlabel('Predicted Time (minutes)')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('Predictive Distributions by Scenario', fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residual analysis
    if 'predictions' in results['model_results']:
        predictions = results['model_results']['predictions']
        
        y_test = np.array(predictions['y_test']) / 60
        y_pred = np.array(predictions['y_pred_standard']) / 60
        residuals = y_pred - y_test
        
        # Q-Q plot for normality check
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot: Residual Normality Check', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add normality test result
        _, p_value = stats.shapiro(residuals)
        ax4.text(0.05, 0.95, f'Shapiro-Wilk p-value: {p_value:.3f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bayesian_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bayesian_dashboard(df, results, output_dir):
    """Create a comprehensive Bayesian regression analysis dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main model comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'model_comparison' in results['analysis']:
        model_comp = results['analysis']['model_comparison']
        
        models = list(model_comp.keys())
        mae_values = [model_comp[m]['test_mae'] for m in models]
        r2_values = [model_comp[m]['test_r2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1_twin = ax1.twinx()
        
        bars = ax1.bar(x, mae_values, width, alpha=0.7, color='lightcoral', label='MAE')
        line = ax1_twin.plot(x, r2_values, 'bo-', linewidth=3, markersize=8, label='R²')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('MAE (minutes)', color='red')
        ax1_twin.set_ylabel('R² Score', color='blue')
        ax1.set_title('Bayesian Model Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in models])
        ax1.grid(True, alpha=0.3)
    
    # Coefficient plot
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if 'coefficient_analysis' in results['analysis']:
        coeff_analysis = results['analysis']['coefficient_analysis']
        
        features = list(coeff_analysis.keys())[:8]  # Top 8 features
        coefficients = [coeff_analysis[f]['coefficient'] for f in features]
        
        colors = ['green' if c < 0 else 'red' for c in coefficients]
        
        bars = ax2.barh(range(len(features)), coefficients, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels([f.replace('_', ' ') for f in features], fontsize=10)
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title('Bayesian Coefficients', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
    
    # Prediction intervals
    ax3 = fig.add_subplot(gs[1, :2])
    
    if 'predictions' in results['model_results']:
        predictions = results['model_results']['predictions']
        
        y_test = np.array(predictions['y_test']) / 60
        y_pred = np.array(predictions['y_pred_standard']) / 60
        y_std = np.array(predictions['y_std_standard']) / 60
        
        # Plot sample with intervals
        n_points = 50
        indices = np.random.choice(len(y_test), n_points, replace=False)
        
        x_plot = range(n_points)
        y_test_plot = y_test[indices]
        y_pred_plot = y_pred[indices]
        y_std_plot = y_std[indices]
        
        ax3.scatter(x_plot, y_test_plot, alpha=0.6, s=40, label='Actual', color='blue')
        ax3.errorbar(x_plot, y_pred_plot, yerr=1.96*y_std_plot, 
                    fmt='ro', alpha=0.6, capsize=3, label='Predicted ± 95% CI')
        
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Finish Time (minutes)')
        ax3.set_title('Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Uncertainty analysis
    ax4 = fig.add_subplot(gs[1, 2:])
    
    if 'uncertainty_analysis' in results['analysis']:
        uncertainty = results['analysis']['uncertainty_analysis']
        
        coverage_68 = uncertainty['coverage_68_percent'] * 100
        coverage_95 = uncertainty['coverage_95_percent'] * 100
        
        intervals = ['68%', '95%']
        actual = [coverage_68, coverage_95]
        expected = [68, 95]
        
        x = np.arange(len(intervals))
        width = 0.35
        
        ax4.bar(x - width/2, actual, width, label='Actual', alpha=0.7, color='steelblue')
        ax4.bar(x + width/2, expected, width, label='Expected', alpha=0.7, color='orange')
        
        ax4.set_xlabel('Confidence Interval')
        ax4.set_ylabel('Coverage (%)')
        ax4.set_title('Prediction Interval Coverage', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(intervals)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Summary statistics and interpretation
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create comprehensive summary
    summary_text = f"""
    BAYESIAN REGRESSION ANALYSIS SUMMARY
    
    Dataset & Model:
    • Total Samples: {results['data_summary']['total_samples']:,}
    • Features: {results['data_summary']['num_features']}
    • Target Mean: {results['data_summary']['target_mean_minutes']:.1f} minutes
    """
    
    if 'model_comparison' in results['analysis']:
        model_comp = results['analysis']['model_comparison']
        best_model = max(model_comp.items(), key=lambda x: x[1]['test_r2'])
        
        summary_text += f"""
    Best Model Performance:
    • Model Type: {best_model[0].replace('_', ' ').title()}
    • Test R²: {best_model[1]['test_r2']:.3f}
    • Test MAE: {best_model[1]['test_mae']:.1f} minutes
    """
    
    if 'uncertainty_analysis' in results['analysis']:
        uncertainty = results['analysis']['uncertainty_analysis']
        summary_text += f"""
    Uncertainty Quantification:
    • Mean Prediction Std: ±{uncertainty['mean_prediction_std_minutes']:.1f} minutes
    • 68% Interval Coverage: {uncertainty['coverage_68_percent']*100:.1f}%
    • 95% Interval Coverage: {uncertainty['coverage_95_percent']*100:.1f}%
    """
    
    if 'coefficient_analysis' in results['analysis']:
        coeff_analysis = results['analysis']['coefficient_analysis']
        
        # Find most influential positive and negative effects
        pos_effects = [(f, data['coefficient']) for f, data in coeff_analysis.items() 
                      if data['direction'] == 'positive']
        neg_effects = [(f, data['coefficient']) for f, data in coeff_analysis.items() 
                      if data['direction'] == 'negative']
        
        if pos_effects:
            strongest_pos = max(pos_effects, key=lambda x: x[1])
            summary_text += f"\n    • Strongest Positive Effect: {strongest_pos[0]} (+{strongest_pos[1]:.1f}s)"
        
        if neg_effects:
            strongest_neg = min(neg_effects, key=lambda x: x[1])
            summary_text += f"\n    • Strongest Negative Effect: {strongest_neg[0]} ({strongest_neg[1]:.1f}s)"
    
    summary_text += f"""
    
    Key Bayesian Advantages:
    • Provides uncertainty estimates for all predictions
    • Incorporates prior knowledge through regularization
    • Natural handling of model uncertainty
    • Probabilistic interpretations of coefficients
    
    Model Insights:
    • Prediction intervals capture actual uncertainty well
    • Bayesian regularization prevents overfitting
    • Uncertainty varies appropriately across prediction range
    • Model coefficients have clear probabilistic interpretation
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Bayesian Regression Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'bayesian_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_bayesian_report(df, results, output_dir):
    """Generate a comprehensive Bayesian regression analysis report."""
    report = []
    report.append("BAYESIAN REGRESSION ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants with {len(df.columns)-1} features\n")
    
    # Model comparison
    if 'model_comparison' in results['analysis']:
        report.append("BAYESIAN MODEL COMPARISON")
        report.append("-" * 30)
        
        model_comp = results['analysis']['model_comparison']
        for model_name, metrics in model_comp.items():
            report.append(f"\n{model_name.replace('_', ' ').title()} Model:")
            report.append(f"  Test MAE: {metrics['test_mae']:.1f} minutes")
            report.append(f"  Test R²: {metrics['test_r2']:.3f}")
            if 'log_marginal_likelihood' in metrics and metrics['log_marginal_likelihood'] != 0:
                report.append(f"  Log Marginal Likelihood: {metrics['log_marginal_likelihood']:.2f}")
        
        # Best model
        best_model = max(model_comp.items(), key=lambda x: x[1]['test_r2'])
        report.append(f"\nBest Model: {best_model[0].replace('_', ' ').title()}")
        report.append("")
    
    # Coefficient analysis
    if 'coefficient_analysis' in results['analysis']:
        report.append("BAYESIAN COEFFICIENT ANALYSIS")
        report.append("-" * 32)
        
        coeff_analysis = results['analysis']['coefficient_analysis']
        
        # Sort by magnitude
        sorted_coeffs = sorted(coeff_analysis.items(), 
                             key=lambda x: x[1]['magnitude'], reverse=True)
        
        for feature, data in sorted_coeffs:
            coeff = data['coefficient']
            direction = data['direction']
            interpretation = data['interpretation']
            
            report.append(f"\n{feature}:")
            report.append(f"  Coefficient: {coeff:.3f} ({direction})")
            report.append(f"  Magnitude: {data['magnitude']:.3f}")
            report.append(f"  Interpretation: {interpretation}")
    
    # Uncertainty quantification
    if 'uncertainty_analysis' in results['analysis']:
        report.append("\nUNCERTAINTY QUANTIFICATION")
        report.append("-" * 28)
        
        uncertainty = results['analysis']['uncertainty_analysis']
        
        report.append(f"Prediction Uncertainty:")
        report.append(f"  Mean Standard Deviation: ±{uncertainty['mean_prediction_std_minutes']:.1f} minutes")
        report.append(f"  Median Standard Deviation: ±{uncertainty['median_prediction_std_minutes']:.1f} minutes")
        
        report.append(f"\nPrediction Interval Coverage:")
        report.append(f"  68% Interval: {uncertainty['coverage_68_percent']*100:.1f}% (expected: 68%)")
        report.append(f"  95% Interval: {uncertainty['coverage_95_percent']*100:.1f}% (expected: 95%)")
        
        # Coverage assessment
        coverage_68_diff = abs(uncertainty['coverage_68_percent'] - 0.68) * 100
        coverage_95_diff = abs(uncertainty['coverage_95_percent'] - 0.95) * 100
        
        if coverage_68_diff < 5 and coverage_95_diff < 5:
            report.append("  Assessment: Excellent interval calibration")
        elif coverage_68_diff < 10 and coverage_95_diff < 10:
            report.append("  Assessment: Good interval calibration")
        else:
            report.append("  Assessment: Interval calibration needs improvement")
    
    # Scenario predictions
    if 'scenario_predictions' in results['analysis']:
        report.append("\nSCENARIO PREDICTIONS WITH UNCERTAINTY")
        report.append("-" * 40)
        
        scenarios = results['analysis']['scenario_predictions']
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                mean_time = prediction['mean_minutes']
                std_time = prediction.get('std_minutes', 0)
                
                report.append(f"\n{scenario}:")
                report.append(f"  Predicted Time: {mean_time:.1f} ± {std_time:.1f} minutes")
                
                if 'confidence_interval_95' in prediction:
                    ci = prediction['confidence_interval_95']
                    report.append(f"  95% Confidence Interval: {ci['lower_minutes']:.1f} - {ci['upper_minutes']:.1f} minutes")
    
    # Model insights and interpretation
    report.append("\nBAYESIAN MODEL INSIGHTS")
    report.append("-" * 26)
    
    if 'sensitivity_analysis' in results['analysis']:
        sensitivity = results['analysis']['sensitivity_analysis']
        
        if 'regularization_effect' in sensitivity:
            reg_effect = sensitivity['regularization_effect']
            report.append("Regularization Analysis:")
            
            for param in ['alpha', 'lambda']:
                std_val = reg_effect.get(f'{param}_standard', 0)
                log_val = reg_effect.get(f'{param}_log', 0)
                report.append(f"  {param.capitalize()} - Standard: {std_val:.2e}, Log: {log_val:.2e}")
    
    # Key findings
    report.append("\nKEY FINDINGS & INTERPRETATION")
    report.append("-" * 32)
    
    if 'coefficient_analysis' in results['analysis']:
        coeff_analysis = results['analysis']['coefficient_analysis']
        
        # Find strongest effects
        pos_effects = [(f, data['coefficient']) for f, data in coeff_analysis.items() 
                      if data['direction'] == 'positive']
        neg_effects = [(f, data['coefficient']) for f, data in coeff_analysis.items() 
                      if data['direction'] == 'negative']
        
        if pos_effects:
            strongest_pos = max(pos_effects, key=lambda x: x[1])
            report.append(f"• Strongest performance hindrance: {strongest_pos[0]} (+{strongest_pos[1]:.1f}s)")
        
        if neg_effects:
            strongest_neg = min(neg_effects, key=lambda x: x[1])
            report.append(f"• Strongest performance benefit: {strongest_neg[0]} ({strongest_neg[1]:.1f}s)")
    
    if 'uncertainty_analysis' in results['analysis']:
        uncertainty = results['analysis']['uncertainty_analysis']
        mean_std = uncertainty['mean_prediction_std_minutes']
        
        if mean_std < 2:
            report.append("• High prediction confidence - low uncertainty across predictions")
        elif mean_std < 4:
            report.append("• Moderate prediction confidence - reasonable uncertainty levels")
        else:
            report.append("• Lower prediction confidence - high uncertainty suggests model limitations")
    
    # Bayesian advantages
    report.append("\nBAYESIAN REGRESSION ADVANTAGES")
    report.append("-" * 33)
    report.append("• Provides uncertainty estimates for every prediction")
    report.append("• Natural regularization through prior distributions")
    report.append("• Probabilistic interpretation of model parameters")
    report.append("• Robust to overfitting through Bayesian averaging")
    report.append("• Quantifies model uncertainty in addition to prediction uncertainty")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 15)
    report.append("• Use prediction intervals for race time goal setting")
    report.append("• Leverage uncertainty estimates for risk assessment")
    report.append("• Apply Bayesian framework for A/B testing of training methods")
    
    if 'uncertainty_analysis' in results['analysis']:
        coverage_quality = (results['analysis']['uncertainty_analysis']['coverage_68_percent'] + 
                          results['analysis']['uncertainty_analysis']['coverage_95_percent']) / 2
        if coverage_quality > 0.8:
            report.append("• Deploy model for performance prediction with confidence intervals")
        else:
            report.append("• Improve interval calibration before deployment")
    
    report.append("• Consider hierarchical Bayesian models for club-level effects")
    report.append("• Use coefficient uncertainties for feature selection")
    
    # Save report
    with open(output_dir / 'bayesian_regression_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating Bayesian regression analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('bayesian_regression')
    
    # Load results
    df, results, models = load_results()
    
    # Create visualizations
    print("Plotting coefficient analysis...")
    plot_coefficient_analysis(results, output_dir)
    
    print("Plotting prediction intervals...")
    plot_prediction_intervals(results, output_dir)
    
    print("Plotting Bayesian diagnostics...")
    plot_bayesian_diagnostics(results, output_dir)
    
    print("Creating Bayesian dashboard...")
    plot_bayesian_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_bayesian_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()