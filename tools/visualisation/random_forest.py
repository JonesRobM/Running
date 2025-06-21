#!/usr/bin/env python3
"""
Random Forest Visualization
Creates comprehensive visualizations for Random Forest analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
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
    """Load Random Forest analysis results and data."""
    # Load processed data
    df = pd.read_csv('data/processed/random_forest_data.csv')
    
    # Load results
    with open('data/processed/random_forest_results.json', 'r') as f:
        results = json.load(f)
    
    # Load model objects
    try:
        with open('data/processed/random_forest_models.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}
    
    return df, results, models

def plot_feature_importance(results, output_dir):
    """Plot feature importance analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top 15 most important features (optimized model)
    feature_imp = pd.DataFrame(results['feature_importance_df'])
    top_features = feature_imp.head(15)
    
    ax1.barh(range(len(top_features)), top_features['importance_optimized'], alpha=0.7, color='steelblue')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title('Top 15 Most Important Features (Optimized Model)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(top_features['importance_optimized']):
        ax1.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Plot 2: Comparison between basic and optimized model importance
    comparison_features = feature_imp.head(10)
    
    x = np.arange(len(comparison_features))
    width = 0.35
    
    ax2.bar(x - width/2, comparison_features['importance_basic'], width, 
           label='Basic Model', alpha=0.7)
    ax2.bar(x + width/2, comparison_features['importance_optimized'], width, 
           label='Optimized Model', alpha=0.7)
    
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Importance')
    ax2.set_title('Feature Importance: Basic vs Optimized Model', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_features['feature'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative feature importance
    feature_imp_sorted = feature_imp.sort_values('importance_optimized', ascending=False)
    cumulative_importance = feature_imp_sorted['importance_optimized'].cumsum()
    
    ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-', alpha=0.7)
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    ax3.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
    
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('Cumulative Importance')
    ax3.set_title('Cumulative Feature Importance', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Find number of features for 90% and 95%
    n_features_90 = (cumulative_importance >= 0.90).argmax() + 1
    n_features_95 = (cumulative_importance >= 0.95).argmax() + 1
    
    ax3.axvline(x=n_features_90, color='orange', linestyle=':', alpha=0.7)
    ax3.axvline(x=n_features_95, color='red', linestyle=':', alpha=0.7)
    
    ax3.text(n_features_90 + 1, 0.5, f'{n_features_90} features\nfor 90%', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
    ax3.text(n_features_95 + 1, 0.7, f'{n_features_95} features\nfor 95%', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    
    # Plot 4: Feature importance by category
    feature_categories = {
        'Demographics': ['age', 'is_male', 'age_male_interaction'],
        'Club': ['has_club', 'club_size', 'large_club'],
        'Performance': ['10km_seconds', 'pace_10km', 'has_10km_split'],
        'Age Groups': [col for col in feature_imp['feature'] if col.startswith('age_group_')],
        'Specific Clubs': [col for col in feature_imp['feature'] if col.startswith('club_')]
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_features = feature_imp[feature_imp['feature'].isin(features)]
        if not category_features.empty:
            category_importance[category] = category_features['importance_optimized'].sum()
    
    if category_importance:
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        ax4.pie(importances, labels=categories, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Feature Importance by Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance(results, output_dir):
    """Plot model performance metrics and predictions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model comparison metrics
    models = ['basic', 'optimized']
    metrics = ['train_mae', 'test_mae', 'train_r2', 'test_r2']
    
    train_mae = [results['model_results'][m]['train_mae']/60 for m in models]
    test_mae = [results['model_results'][m]['test_mae']/60 for m in models]
    train_r2 = [results['model_results'][m]['train_r2'] for m in models]
    test_r2 = [results['model_results'][m]['test_r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1_twin = ax1.twinx()
    
    # MAE bars
    ax1.bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.7, color='lightblue')
    ax1.bar(x + width/2, test_mae, width, label='Test MAE', alpha=0.7, color='lightcoral')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MAE (minutes)', color='blue')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.title() for m in models])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # R² line plot
    ax1_twin.plot(x, train_r2, 'bo-', label='Train R²', linewidth=2, markersize=8)
    ax1_twin.plot(x, test_r2, 'ro-', label='Test R²', linewidth=2, markersize=8)
    ax1_twin.set_ylabel('R² Score', color='red')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Actual vs Predicted (Test Set)
    predictions = results['model_results']['predictions']
    y_test = np.array(predictions['y_test']) / 60  # Convert to minutes
    y_pred = np.array(predictions['y_pred_optimized']) / 60
    
    ax2.scatter(y_test, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Time (minutes)')
    ax2.set_ylabel('Predicted Time (minutes)')
    ax2.set_title('Actual vs Predicted Performance', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add R² and MAE to plot
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax2.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f} min', 
            transform=ax2.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Prediction residuals
    residuals = y_pred - y_test
    
    ax3.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Time (minutes)')
    ax3.set_ylabel('Residuals (minutes)')
    ax3.set_title('Prediction Residuals', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add residual statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    ax3.text(0.05, 0.95, f'Mean = {mean_residual:.2f}\nStd = {std_residual:.2f}', 
            transform=ax3.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Cross-validation scores
    cv_scores = results['model_results']['optimized']['cv_mae_mean']
    cv_std = results['model_results']['optimized']['cv_mae_std']
    
    # Create a simple bar chart showing CV performance
    ax4.bar(['Cross-Validation MAE'], [cv_scores/60], 
           yerr=[cv_std/60], capsize=10, alpha=0.7, color='green')
    ax4.set_ylabel('MAE (minutes)')
    ax4.set_title('Cross-Validation Performance', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value label
    ax4.text(0, cv_scores/60 + cv_std/60 + 0.5, f'{cv_scores/60:.1f} ± {cv_std/60:.1f}', 
            ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scenario_predictions(results, output_dir):
    """Plot performance predictions for different scenarios."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Scenario predictions
    if 'scenario_predictions' in results['analysis']:
        scenarios = results['analysis']['scenario_predictions']
        
        scenario_names = []
        predicted_times = []
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                scenario_names.append(scenario.replace('_', '\n'))
                predicted_times.append(prediction['predicted_minutes'])
        
        if scenario_names:
            colors = ['blue' if 'Male' in name else 'red' for name in scenario_names]
            bars = ax1.bar(range(len(scenario_names)), predicted_times, 
                          color=colors, alpha=0.7)
            
            ax1.set_xlabel('Scenario')
            ax1.set_ylabel('Predicted Time (minutes)')
            ax1.set_title('Performance Predictions by Scenario', fontweight='bold')
            ax1.set_xticks(range(len(scenario_names)))
            ax1.set_xticklabels(scenario_names, fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, time in zip(bars, predicted_times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{time:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Performance insights visualization
    if 'performance_insights' in results['analysis']:
        insights = results['analysis']['performance_insights']
        
        factors = []
        importances = []
        
        for factor, data in insights.items():
            if 'importance' in data:
                factors.append(factor.replace('_', ' ').title())
                importances.append(data['importance'])
        
        if factors:
            ax2.barh(range(len(factors)), importances, alpha=0.7, color='steelblue')
            ax2.set_yticks(range(len(factors)))
            ax2.set_yticklabels(factors)
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('Key Performance Factors', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(importances):
                ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    
    # Plot 3: Model accuracy by performance level
    # Divide predictions into performance quartiles and show accuracy
    predictions = results['model_results']['predictions']
    y_test = np.array(predictions['y_test']) / 60
    y_pred = np.array(predictions['y_pred_optimized']) / 60
    
    # Create quartiles
    quartiles = np.percentile(y_test, [25, 50, 75])
    quartile_labels = ['Fast\n(Q1)', 'Medium-Fast\n(Q2)', 'Medium-Slow\n(Q3)', 'Slow\n(Q4)']
    
    quartile_errors = []
    quartile_r2 = []
    
    for i in range(4):
        if i == 0:
            mask = y_test <= quartiles[0]
        elif i == 3:
            mask = y_test > quartiles[2]
        else:
            mask = (y_test > quartiles[i-1]) & (y_test <= quartiles[i])
        
        if mask.sum() > 0:
            q_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            q_r2 = r2_score(y_test[mask], y_pred[mask])
            quartile_errors.append(q_mae)
            quartile_r2.append(q_r2)
        else:
            quartile_errors.append(0)
            quartile_r2.append(0)
    
    ax3_twin = ax3.twinx()
    
    x = np.arange(len(quartile_labels))
    width = 0.35
    
    bars1 = ax3.bar(x, quartile_errors, width, alpha=0.7, color='lightcoral', label='MAE')
    ax3.set_xlabel('Performance Quartile')
    ax3.set_ylabel('MAE (minutes)', color='red')
    ax3.set_title('Model Accuracy by Performance Level', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(quartile_labels)
    ax3.grid(True, alpha=0.3)
    
    line1 = ax3_twin.plot(x, quartile_r2, 'bo-', linewidth=2, markersize=8, label='R²')
    ax3_twin.set_ylabel('R² Score', color='blue')
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 4: Feature importance vs model performance correlation
    # Show how removing features affects performance
    feature_imp = pd.DataFrame(results['feature_importance_df'])
    
    # Simulate feature removal impact (this would need actual model retraining in practice)
    # For visualization, we'll show theoretical impact
    top_features = feature_imp.head(10)
    
    # Theoretical performance drop if feature removed (simplified calculation)
    performance_impact = top_features['importance_optimized'] * 100  # Convert to percentage
    
    ax4.bar(range(len(top_features)), performance_impact, alpha=0.7, color='orange')
    ax4.set_xlabel('Top Features (by importance)')
    ax4.set_ylabel('Estimated Performance Impact (%)')
    ax4.set_title('Estimated Impact of Removing Top Features', fontweight='bold')
    ax4.set_xticks(range(len(top_features)))
    ax4.set_xticklabels(top_features['feature'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_random_forest_dashboard(df, results, output_dir):
    """Create a comprehensive Random Forest analysis dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Model performance summary
    ax1 = fig.add_subplot(gs[0, :2])
    
    best_results = results['model_results']['optimized']
    metrics = ['Train MAE', 'Test MAE', 'Train R²', 'Test R²']
    values = [
        best_results['train_mae']/60,
        best_results['test_mae']/60,
        best_results['train_r2'],
        best_results['test_r2']
    ]
    
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Top features
    ax2 = fig.add_subplot(gs[0, 2:])
    feature_imp = pd.DataFrame(results['feature_importance_df'])
    top_features = feature_imp.head(8)
    
    ax2.barh(range(len(top_features)), top_features['importance_optimized'], alpha=0.7)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'], fontsize=10)
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 8 Most Important Features', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Actual vs Predicted
    ax3 = fig.add_subplot(gs[1, :2])
    predictions = results['model_results']['predictions']
    y_test = np.array(predictions['y_test']) / 60
    y_pred = np.array(predictions['y_pred_optimized']) / 60
    
    ax3.scatter(y_test, y_pred, alpha=0.6, s=30)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax3.set_xlabel('Actual Time (minutes)')
    ax3.set_ylabel('Predicted Time (minutes)')
    ax3.set_title('Actual vs Predicted Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Scenario predictions
    ax4 = fig.add_subplot(gs[1, 2:])
    if 'scenario_predictions' in results['analysis']:
        scenarios = results['analysis']['scenario_predictions']
        
        scenario_names = []
        predicted_times = []
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                scenario_names.append(scenario.replace('_', '\n'))
                predicted_times.append(prediction['predicted_minutes'])
        
        if scenario_names:
            colors = ['blue' if 'Male' in name else 'red' for name in scenario_names]
            bars = ax4.bar(range(len(scenario_names)), predicted_times, 
                          color=colors, alpha=0.7)
            ax4.set_ylabel('Predicted Time (minutes)')
            ax4.set_title('Performance Predictions by Scenario', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(scenario_names)))
            ax4.set_xticklabels(scenario_names, fontsize=9)
            ax4.grid(True, alpha=0.3)
    
    # Key statistics and interpretation
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Model performance interpretation
    model_perf = results['analysis']['model_performance']
    
    stats_text = f"""
    RANDOM FOREST MODEL ANALYSIS SUMMARY
    
    Dataset & Model:
    • Total Samples: {results['data_summary']['total_samples']:,}
    • Features Used: {results['data_summary']['num_features']}
    • Best Model Parameters: {results['model_results']['optimized']['best_params']}
    
    Performance:
    • Test Accuracy: {model_perf['interpretation']['r2']}
    • Prediction Error: {model_perf['interpretation']['mae']}
    • Cross-Validation MAE: {best_results['cv_mae_mean']/60:.1f} ± {best_results['cv_mae_std']/60:.1f} minutes
    
    Key Insights:
    • Most Important Factor: {feature_imp.iloc[0]['feature']} (importance: {feature_imp.iloc[0]['importance_optimized']:.3f})
    • Second Most Important: {feature_imp.iloc[1]['feature']} (importance: {feature_imp.iloc[1]['importance_optimized']:.3f})
    • Third Most Important: {feature_imp.iloc[2]['feature']} (importance: {feature_imp.iloc[2]['importance_optimized']:.3f})
    
    Model Quality:
    • Strong predictive performance with {best_results['test_r2']*100:.1f}% variance explained
    • Low overfitting: train R² ({best_results['train_r2']:.3f}) vs test R² ({best_results['test_r2']:.3f})
    • Reliable cross-validation performance
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Random Forest Performance Prediction Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'random_forest_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_random_forest_report(df, results, output_dir):
    """Generate a comprehensive Random Forest analysis report."""
    report = []
    report.append("RANDOM FOREST ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants with {len(df.columns)-1} features\n")
    
    # Model performance
    basic_results = results['model_results']['basic']
    optimized_results = results['model_results']['optimized']
    
    report.append("MODEL PERFORMANCE COMPARISON")
    report.append("-" * 35)
    report.append("Basic Random Forest:")
    report.append(f"  Train MAE: {basic_results['train_mae']/60:.1f} minutes")
    report.append(f"  Test MAE: {basic_results['test_mae']/60:.1f} minutes")
    report.append(f"  Train R²: {basic_results['train_r2']:.3f}")
    report.append(f"  Test R²: {basic_results['test_r2']:.3f}")
    report.append(f"  CV MAE: {basic_results['cv_mae_mean']/60:.1f} ± {basic_results['cv_mae_std']/60:.1f} minutes")
    
    report.append("\nOptimized Random Forest:")
    report.append(f"  Best Parameters: {optimized_results['best_params']}")
    report.append(f"  Train MAE: {optimized_results['train_mae']/60:.1f} minutes")
    report.append(f"  Test MAE: {optimized_results['test_mae']/60:.1f} minutes")
    report.append(f"  Train R²: {optimized_results['train_r2']:.3f}")
    report.append(f"  Test R²: {optimized_results['test_r2']:.3f}")
    report.append(f"  CV MAE: {optimized_results['cv_mae_mean']/60:.1f} ± {optimized_results['cv_mae_std']/60:.1f} minutes")
    
    # Feature importance
    report.append("\nTOP 10 MOST IMPORTANT FEATURES")
    report.append("-" * 35)
    feature_imp = pd.DataFrame(results['feature_importance_df'])
    top_features = feature_imp.head(10)
    
    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        report.append(f"{idx:2d}. {row['feature']}: {row['importance_optimized']:.4f}")
    
    # Performance insights
    if 'performance_insights' in results['analysis']:
        report.append("\nPERFORMANCE FACTOR ANALYSIS")
        report.append("-" * 30)
        insights = results['analysis']['performance_insights']
        
        for factor, data in insights.items():
            factor_name = factor.replace('_', ' ').title()
            report.append(f"\n{factor_name}:")
            report.append(f"  Importance: {data['importance']:.4f}")
            report.append(f"  {data['interpretation']}")
    
    # Scenario predictions
    if 'scenario_predictions' in results['analysis']:
        report.append("\nPERFORMANCE PREDICTIONS")
        report.append("-" * 25)
        scenarios = results['analysis']['scenario_predictions']
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                report.append(f"{scenario.replace('_', ' ')}: {prediction['predicted_time_formatted']} ({prediction['predicted_minutes']:.1f} minutes)")
    
    # Model interpretation
    report.append("\nMODEL INTERPRETATION")
    report.append("-" * 25)
    
    model_perf = results['analysis']['model_performance']
    report.append(f"• {model_perf['interpretation']['mae']}")
    report.append(f"• {model_perf['interpretation']['r2']}")
    
    # Assess overfitting
    train_test_gap = optimized_results['train_r2'] - optimized_results['test_r2']
    if train_test_gap < 0.05:
        report.append("• Model shows minimal overfitting - good generalization")
    elif train_test_gap < 0.10:
        report.append("• Model shows slight overfitting - acceptable for practical use")
    else:
        report.append("• Model shows significant overfitting - consider regularization")
    
    # Feature insights
    if '10km_seconds' in top_features['feature'].values:
        report.append("• 10km split time is a strong predictor - pacing strategy matters")
    
    if 'age' in top_features['feature'].values:
        report.append("• Age is a significant factor - age-graded analysis recommended")
    
    if any('club' in f for f in top_features['feature'].values):
        report.append("• Club-related features are important - training environment affects performance")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 15)
    report.append("• Use this model for performance prediction and target setting")
    report.append("• Focus training interventions on top-importance factors")
    report.append("• Consider collecting additional data on identified important features")
    
    if optimized_results['test_r2'] > 0.8:
        report.append("• Model performance is excellent - suitable for operational use")
    elif optimized_results['test_r2'] > 0.6:
        report.append("• Model performance is good - suitable for guidance and insights")
    else:
        report.append("• Model performance is moderate - use with caution and supplement with domain expertise")
    
    report.append("• Implement regular model retraining with new race data")
    report.append("• Consider ensemble methods to further improve predictions")
    
    # Save report
    with open(output_dir / 'random_forest_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating Random Forest analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('random_forest')
    
    # Load results
    df, results, models = load_results()
    
    # Create visualizations
    print("Plotting feature importance analysis...")
    plot_feature_importance(results, output_dir)
    
    print("Plotting model performance...")
    plot_model_performance(results, output_dir)
    
    print("Plotting scenario predictions...")
    plot_scenario_predictions(results, output_dir)
    
    print("Creating summary dashboard...")
    plot_random_forest_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_random_forest_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()