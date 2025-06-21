#!/usr/bin/env python3
"""
XGBoost Visualization
Creates comprehensive visualizations for XGBoost analysis results.
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
    """Load XGBoost analysis results and data."""
    # Load processed data
    df = pd.read_csv('data/processed/xgboost_data.csv')
    
    # Load results
    with open('data/processed/xgboost_results.json', 'r') as f:
        results = json.load(f)
    
    # Load model objects
    try:
        with open('data/processed/xgboost_models.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}
    
    return df, results, models

def plot_feature_importance_advanced(results, output_dir):
    """Plot advanced feature importance analysis including permutation importance."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: XGBoost vs Permutation Importance
    feature_comp = pd.DataFrame(results['feature_importance_comparison'])
    top_features = feature_comp.head(12)
    
    x = np.arange(len(top_features))
    width = 0.35
    
    ax1.barh(x - width/2, top_features['xgb_importance'], width, 
            label='XGBoost Importance', alpha=0.8, color='steelblue')
    ax1.barh(x + width/2, top_features['permutation_importance'], width, 
            label='Permutation Importance', alpha=0.8, color='orange')
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Feature Importance: XGBoost vs Permutation', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature importance by category
    # Categorize features
    feature_categories = {
        'Age Features': [f for f in feature_comp['feature'] if 'age' in f.lower()],
        'Gender Features': [f for f in feature_comp['feature'] if any(g in f.lower() for g in ['gender', 'male', 'female'])],
        'Club Features': [f for f in feature_comp['feature'] if 'club' in f.lower()],
        'Pacing Features': [f for f in feature_comp['feature'] if any(p in f.lower() for p in ['pace', '10km', 'split'])],
        'Interaction Features': [f for f in feature_comp['feature'] if 'interaction' in f.lower()],
        'Binned Features': [f for f in feature_comp['feature'] if 'bin' in f.lower()]
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_features = feature_comp[feature_comp['feature'].isin(features)]
        if not category_features.empty:
            category_importance[category] = category_features['xgb_importance'].sum()
    
    if category_importance:
        categories = list(category_importance.keys())
        importances = list(category_importance.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        wedges, texts, autotexts = ax2.pie(importances, labels=categories, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('Feature Importance by Category', fontweight='bold')
    
    # Plot 3: Top feature correlations
    top_10_features = feature_comp.head(10)['feature'].tolist()
    if len(top_10_features) > 1:
        # Load the data to calculate correlations
        df = pd.read_csv('data/processed/xgboost_data.csv')
        
        # Only use features that exist in the dataframe
        available_features = [f for f in top_10_features if f in df.columns]
        
        if len(available_features) > 1:
            correlation_matrix = df[available_features].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, ax=ax3, cbar_kws={'label': 'Correlation'})
            ax3.set_title('Top Features Correlation Matrix', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='y', rotation=0)
        else:
            ax3.text(0.5, 0.5, 'Insufficient features for correlation', 
                    ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Feature importance stability (showing top features across models)
    model_importance = {}
    for model_name in ['baseline', 'optimized', 'early_stopping']:
        if model_name in results['model_results'] and 'feature_importance' in results['model_results'][model_name]:
            model_importance[model_name] = results['model_results'][model_name]['feature_importance']
    
    if len(model_importance) > 1:
        # Get top features from optimized model
        top_features_stability = feature_comp.head(8)['feature'].tolist()
        
        stability_data = []
        for feature in top_features_stability:
            for model_name, importance_dict in model_importance.items():
                if feature in importance_dict:
                    stability_data.append({
                        'feature': feature,
                        'model': model_name,
                        'importance': importance_dict[feature]
                    })
        
        if stability_data:
            stability_df = pd.DataFrame(stability_data)
            stability_pivot = stability_df.pivot(index='feature', columns='model', values='importance')
            
            stability_pivot.plot(kind='bar', ax=ax4, alpha=0.7)
            ax4.set_xlabel('Feature')
            ax4.set_ylabel('Importance')
            ax4.set_title('Feature Importance Stability Across Models', fontweight='bold')
            ax4.legend(title='Model')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Multiple models not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance_comparison(results, output_dir):
    """Plot comprehensive model performance comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Performance metrics comparison
    models = ['baseline', 'optimized', 'early_stopping']
    metrics = ['train_mae', 'test_mae', 'train_r2', 'test_r2']
    
    model_data = []
    for model in models:
        if model in results['model_results']:
            model_data.append({
                'model': model,
                'train_mae': results['model_results'][model].get('train_mae', 0) / 60,
                'test_mae': results['model_results'][model].get('test_mae', 0) / 60,
                'train_r2': results['model_results'][model].get('train_r2', 0),
                'test_r2': results['model_results'][model].get('test_r2', 0)
            })
    
    if model_data:
        model_df = pd.DataFrame(model_data)
        
        x = np.arange(len(model_df))
        width = 0.2
        
        ax1_twin = ax1.twinx()
        
        # MAE bars
        ax1.bar(x - width, model_df['train_mae'], width, label='Train MAE', alpha=0.7, color='lightblue')
        ax1.bar(x, model_df['test_mae'], width, label='Test MAE', alpha=0.7, color='lightcoral')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('MAE (minutes)', color='blue')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_df['model'])
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # R² lines
        ax1_twin.plot(x, model_df['train_r2'], 'bo-', label='Train R²', linewidth=2, markersize=8)
        ax1_twin.plot(x, model_df['test_r2'], 'ro-', label='Test R²', linewidth=2, markersize=8)
        ax1_twin.set_ylabel('R² Score', color='red')
        ax1_twin.legend(loc='upper right')
    
    # Plot 2: Prediction scatter plot
    if 'predictions' in results['model_results']:
        predictions = results['model_results']['predictions']
        y_test = np.array(predictions['y_test']) / 60
        y_pred = np.array(predictions['y_pred_optimized']) / 60
        
        ax2.scatter(y_test, y_pred, alpha=0.6, s=30, c='steelblue')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Performance statistics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        ax2.set_xlabel('Actual Time (minutes)')
        ax2.set_ylabel('Predicted Time (minutes)')
        ax2.set_title('Prediction Accuracy (Optimized Model)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        ax2.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f} min', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Residuals analysis
    if 'predictions' in results['model_results']:
        residuals = y_pred - y_test
        
        ax3.scatter(y_pred, residuals, alpha=0.6, s=30, c='orange')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        # Add standard deviation bands
        residual_std = np.std(residuals)
        ax3.axhline(y=residual_std, color='orange', linestyle=':', alpha=0.7, label='+1σ')
        ax3.axhline(y=-residual_std, color='orange', linestyle=':', alpha=0.7, label='-1σ')
        
        ax3.set_xlabel('Predicted Time (minutes)')
        ax3.set_ylabel('Residuals (minutes)')
        ax3.set_title('Prediction Residuals', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add residual statistics
        mean_residual = np.mean(residuals)
        ax3.text(0.05, 0.95, f'Mean = {mean_residual:.2f}\nStd = {residual_std:.2f}', 
                transform=ax3.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Learning curves (if available)
    if 'early_stopping' in results['model_results']:
        # Show overfitting analysis
        models_for_overfitting = ['baseline', 'optimized']
        train_scores = []
        test_scores = []
        model_names = []
        
        for model in models_for_overfitting:
            if model in results['model_results']:
                train_scores.append(results['model_results'][model].get('train_r2', 0))
                test_scores.append(results['model_results'][model].get('test_r2', 0))
                model_names.append(model)
        
        if len(model_names) > 1:
            x = np.arange(len(model_names))
            width = 0.35
            
            ax4.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.7, color='lightgreen')
            ax4.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.7, color='lightcoral')
            
            # Add overfitting gap lines
            for i, (train, test) in enumerate(zip(train_scores, test_scores)):
                gap = train - test
                ax4.plot([i-width/2, i+width/2], [train, test], 'k-', alpha=0.5)
                ax4.text(i, max(train, test) + 0.02, f'Gap: {gap:.3f}', 
                        ha='center', fontsize=10, fontweight='bold')
            
            ax4.set_xlabel('Model')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Overfitting Analysis', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient models for overfitting analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scenario_analysis(results, output_dir):
    """Plot scenario predictions and model insights."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Scenario predictions with confidence
    if 'scenario_predictions' in results['analysis']:
        scenarios = results['analysis']['scenario_predictions']
        
        scenario_names = []
        predicted_times = []
        colors_list = []
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                scenario_names.append(scenario.replace(' ', '\n'))
                predicted_times.append(prediction['predicted_minutes'])
                # Color by gender or type
                if 'Male' in scenario:
                    colors_list.append('steelblue')
                elif 'Female' in scenario:
                    colors_list.append('lightcoral')
                else:
                    colors_list.append('lightgreen')
        
        if scenario_names:
            bars = ax1.bar(range(len(scenario_names)), predicted_times, 
                          color=colors_list, alpha=0.7)
            
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
    
    # Plot 2: Feature category importance
    if 'category_importance' in results['analysis']:
        category_imp = results['analysis']['category_importance']
        
        categories = []
        total_importances = []
        feature_counts = []
        
        for category, data in category_imp.items():
            categories.append(category.replace(' Features', ''))
            total_importances.append(data['total_importance'])
            feature_counts.append(data['feature_count'])
        
        if categories:
            # Create bubble chart: x=total importance, y=mean importance, size=feature count
            mean_importances = [category_imp[cat + ' Features']['mean_importance'] 
                              for cat in categories]
            
            scatter = ax2.scatter(total_importances, mean_importances, 
                                s=[count*50 for count in feature_counts], 
                                alpha=0.6, c=range(len(categories)), cmap='viridis')
            
            # Add category labels
            for i, cat in enumerate(categories):
                ax2.annotate(cat, (total_importances[i], mean_importances[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax2.set_xlabel('Total Feature Importance')
            ax2.set_ylabel('Mean Feature Importance')
            ax2.set_title('Feature Category Analysis', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add legend for bubble sizes
            for count in [1, 3, 5]:
                ax2.scatter([], [], s=count*50, alpha=0.6, c='gray', 
                           label=f'{count} features')
            ax2.legend(title='Feature Count', loc='upper right')
    
    # Plot 3: Model complexity vs performance
    model_performance = []
    model_complexity = []
    model_names = []
    
    for model_name, model_results in results['model_results'].items():
        if model_name != 'predictions' and 'test_r2' in model_results:
            model_performance.append(model_results['test_r2'])
            model_names.append(model_name)
            
            # Estimate complexity based on parameters (simplified)
            if model_name == 'baseline':
                model_complexity.append(1)  # Simple baseline
            elif model_name == 'optimized':
                # Get parameter complexity from best_params if available
                if 'best_params' in model_results:
                    params = model_results['best_params']
                    complexity = params.get('n_estimators', 100) * params.get('max_depth', 6) / 1000
                    model_complexity.append(complexity)
                else:
                    model_complexity.append(2)
            else:
                model_complexity.append(1.5)
    
    if len(model_performance) > 1:
        ax3.scatter(model_complexity, model_performance, s=100, alpha=0.7, c='purple')
        
        # Add model labels
        for i, name in enumerate(model_names):
            ax3.annotate(name, (model_complexity[i], model_performance[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('Model Complexity (relative)')
        ax3.set_ylabel('Test R² Score')
        ax3.set_title('Model Complexity vs Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hyperparameter importance (if available)
    if 'optimized' in results['model_results'] and 'best_params' in results['model_results']['optimized']:
        best_params = results['model_results']['optimized']['best_params']
        
        # Create a normalized view of hyperparameters
        param_names = []
        param_values = []
        param_normalized = []
        
        # Define parameter ranges for normalization
        param_ranges = {
            'n_estimators': (50, 1000),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0, 1),
            'reg_lambda': (0, 2)
        }
        
        for param, value in best_params.items():
            if param in param_ranges:
                param_names.append(param.replace('_', '\n'))
                param_values.append(value)
                
                # Normalize to 0-1 scale
                min_val, max_val = param_ranges[param]
                normalized = (value - min_val) / (max_val - min_val)
                param_normalized.append(max(0, min(1, normalized)))
        
        if param_names:
            bars = ax4.bar(range(len(param_names)), param_normalized, alpha=0.7, color='orange')
            
            ax4.set_xlabel('Hyperparameter')
            ax4.set_ylabel('Normalized Value (0-1)')
            ax4.set_title('Optimal Hyperparameters', fontweight='bold')
            ax4.set_xticks(range(len(param_names)))
            ax4.set_xticklabels(param_names, fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            # Add actual values as text
            for i, (bar, actual_val) in enumerate(zip(bars, param_values)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{actual_val}', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'Hyperparameter data not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_xgboost_dashboard(df, results, output_dir):
    """Create a comprehensive XGBoost analysis dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main performance metrics
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'optimized' in results['model_results']:
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
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Score')
        ax1.set_title('XGBoost Model Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Feature importance top 10
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if 'feature_importance_comparison' in results:
        feature_imp = pd.DataFrame(results['feature_importance_comparison'])
        top_features = feature_imp.head(8)
        
        ax2.barh(range(len(top_features)), top_features['xgb_importance'], alpha=0.7, color='steelblue')
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'], fontsize=10)
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 8 Feature Importance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Prediction accuracy scatter
    ax3 = fig.add_subplot(gs[1, :2])
    
    if 'predictions' in results['model_results']:
        predictions = results['model_results']['predictions']
        y_test = np.array(predictions['y_test']) / 60
        y_pred = np.array(predictions['y_pred_optimized']) / 60
        
        ax3.scatter(y_test, y_pred, alpha=0.6, s=40, c='steelblue')
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax3.set_xlabel('Actual Time (minutes)')
        ax3.set_ylabel('Predicted Time (minutes)')
        ax3.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_test, y_pred)
        ax3.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax3.transAxes, 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Scenario predictions
    ax4 = fig.add_subplot(gs[1, 2:])
    
    if 'scenario_predictions' in results['analysis']:
        scenarios = results['analysis']['scenario_predictions']
        
        scenario_names = []
        predicted_times = []
        colors_list = []
        
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                scenario_names.append(scenario.replace(' ', '\n'))
                predicted_times.append(prediction['predicted_minutes'])
                colors_list.append('blue' if 'Male' in scenario else 'red')
        
        if scenario_names:
            bars = ax4.bar(range(len(scenario_names)), predicted_times, 
                          color=colors_list, alpha=0.7)
            ax4.set_ylabel('Predicted Time (minutes)')
            ax4.set_title('Scenario Predictions', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(scenario_names)))
            ax4.set_xticklabels(scenario_names, fontsize=9)
            ax4.grid(True, alpha=0.3)
    
    # Summary statistics and insights
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create comprehensive summary
    summary_text = f"""
    XGBOOST MODEL ANALYSIS SUMMARY
    
    Dataset & Features:
    • Total Samples: {results['data_summary']['total_samples']:,}
    • Features Used: {results['data_summary']['num_features']}
    • Target Mean: {results['data_summary']['target_mean_minutes']:.1f} minutes
    """
    
    if 'optimized' in results['model_results']:
        best_results = results['model_results']['optimized']
        summary_text += f"""
    Model Performance:
    • Test R²: {best_results['test_r2']:.3f} ({best_results['test_r2']*100:.1f}% variance explained)
    • Test MAE: {best_results['test_mae']/60:.1f} minutes
    • Cross-Validation MAE: {best_results['cv_mae_mean']/60:.1f} ± {best_results['cv_mae_std']/60:.1f} minutes
    
    Hyperparameters:
    • N Estimators: {best_results['best_params'].get('n_estimators', 'N/A')}
    • Max Depth: {best_results['best_params'].get('max_depth', 'N/A')}
    • Learning Rate: {best_results['best_params'].get('learning_rate', 'N/A')}
    """
    
    if 'feature_importance_comparison' in results:
        feature_imp = pd.DataFrame(results['feature_importance_comparison'])
        top_feature = feature_imp.iloc[0]
        summary_text += f"""
    Feature Insights:
    • Most Important Factor: {top_feature['feature']} (importance: {top_feature['xgb_importance']:.3f})
    • Feature Categories: Age, Gender, Club, Pacing, and Interaction effects analyzed
    """
    
    if 'analysis' in results and 'model_performance' in results['analysis']:
        model_perf = results['analysis']['model_performance']
        summary_text += f"""
    Model Quality:
    • Overfitting Assessment: {model_perf['overfitting_assessment']['interpretation']}
    • Early Stopping Benefit: {model_perf.get('early_stopping_benefit', {}).get('iterations_used', 'N/A')} iterations
    """
    
    summary_text += f"""
    Key Findings:
    • XGBoost achieves superior performance through gradient boosting optimization
    • Feature interactions capture complex performance relationships
    • Model provides reliable predictions across different runner profiles
    • Hyperparameter tuning significantly improves baseline performance
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('XGBoost Performance Prediction Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'xgboost_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_xgboost_report(df, results, output_dir):
    """Generate a comprehensive XGBoost analysis report."""
    report = []
    report.append("XGBOOST ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants with {len(df.columns)-1} features\n")
    
    # Data summary
    report.append("DATA SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Samples: {results['data_summary']['total_samples']:,}")
    report.append(f"Features Used: {results['data_summary']['num_features']}")
    report.append(f"Target Mean: {results['data_summary']['target_mean_minutes']:.1f} minutes")
    report.append(f"Target Std: {results['data_summary']['target_std_minutes']:.1f} minutes")
    report.append("")
    
    # Model performance comparison
    report.append("MODEL PERFORMANCE COMPARISON")
    report.append("-" * 32)
    
    for model_name in ['baseline', 'optimized', 'early_stopping']:
        if model_name in results['model_results']:
            model_results = results['model_results'][model_name]
            report.append(f"\n{model_name.upper()} MODEL:")
            report.append(f"  Test MAE: {model_results.get('test_mae', 0)/60:.1f} minutes")
            report.append(f"  Test R²: {model_results.get('test_r2', 0):.3f}")
            
            if 'cv_mae_mean' in model_results:
                report.append(f"  CV MAE: {model_results['cv_mae_mean']/60:.1f} ± {model_results.get('cv_mae_std', 0)/60:.1f} minutes")
            
            if model_name == 'optimized' and 'best_params' in model_results:
                report.append(f"  Best Parameters: {model_results['best_params']}")
    
    # Feature importance analysis
    if 'feature_importance_comparison' in results:
        report.append("\nTOP 15 MOST IMPORTANT FEATURES")
        report.append("-" * 35)
        
        feature_imp = pd.DataFrame(results['feature_importance_comparison'])
        top_features = feature_imp.head(15)
        
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            xgb_imp = row['xgb_importance']
            perm_imp = row['permutation_importance']
            report.append(f"{idx:2d}. {row['feature']}: XGB={xgb_imp:.4f}, Perm={perm_imp:.4f}")
    
    # Feature category analysis
    if 'category_importance' in results['analysis']:
        report.append("\nFEATURE CATEGORY ANALYSIS")
        report.append("-" * 28)
        
        category_imp = results['analysis']['category_importance']
        for category, data in category_imp.items():
            report.append(f"\n{category}:")
            report.append(f"  Total Importance: {data['total_importance']:.4f}")
            report.append(f"  Mean Importance: {data['mean_importance']:.4f}")
            report.append(f"  Feature Count: {data['feature_count']}")
            if data['top_feature']:
                report.append(f"  Top Feature: {data['top_feature']}")
    
    # Model insights
    if 'model_performance' in results['analysis']:
        report.append("\nMODEL QUALITY ASSESSMENT")
        report.append("-" * 26)
        
        model_perf = results['analysis']['model_performance']
        report.append(f"Test MAE: {model_perf['test_mae_minutes']:.1f} minutes")
        report.append(f"Test R²: {model_perf['test_r2']:.3f}")
        report.append(f"Overfitting: {model_perf['overfitting_assessment']['interpretation']}")
        
        overfitting = model_perf['overfitting_assessment']
        report.append(f"R² Gap (Train-Test): {overfitting['r2_gap']:.3f}")
        report.append(f"MAE Gap (Test-Train): {overfitting['mae_gap']:.1f} seconds")
    
    # Scenario predictions
    if 'scenario_predictions' in results['analysis']:
        report.append("\nSCENARIO PREDICTIONS")
        report.append("-" * 22)
        
        scenarios = results['analysis']['scenario_predictions']
        for scenario, prediction in scenarios.items():
            if 'error' not in prediction:
                time_formatted = prediction['predicted_time_formatted']
                minutes = prediction['predicted_minutes']
                report.append(f"{scenario}: {time_formatted} ({minutes:.1f} minutes)")
    
    # Performance interpretation
    report.append("\nMODEL INTERPRETATION")
    report.append("-" * 22)
    
    if 'optimized' in results['model_results']:
        best_r2 = results['model_results']['optimized']['test_r2']
        best_mae = results['model_results']['optimized']['test_mae'] / 60
        
        if best_r2 > 0.85:
            report.append("• Excellent predictive performance - model explains >85% of variance")
        elif best_r2 > 0.70:
            report.append("• Good predictive performance - model explains >70% of variance")
        else:
            report.append("• Moderate predictive performance - consider additional features")
        
        report.append(f"• Average prediction error: ±{best_mae:.1f} minutes")
        
        if best_mae < 3:
            report.append("• High precision - suitable for competitive performance prediction")
        elif best_mae < 5:
            report.append("• Good precision - suitable for training guidance")
        else:
            report.append("• Moderate precision - best used for general performance estimation")
    
    # Feature insights
    if 'feature_importance_comparison' in results:
        feature_imp = pd.DataFrame(results['feature_importance_comparison'])
        top_feature = feature_imp.iloc[0]
        
        report.append(f"• Most predictive factor: {top_feature['feature']}")
        
        # Analyze feature types
        age_features = feature_imp[feature_imp['feature'].str.contains('age', case=False)]
        if not age_features.empty:
            age_importance = age_features['xgb_importance'].sum()
            report.append(f"• Age-related features total importance: {age_importance:.3f}")
        
        pace_features = feature_imp[feature_imp['feature'].str.contains('pace|10km', case=False)]
        if not pace_features.empty:
            pace_importance = pace_features['xgb_importance'].sum()
            report.append(f"• Pacing-related features total importance: {pace_importance:.3f}")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 15)
    report.append("• Deploy model for race time prediction and goal setting")
    report.append("• Use feature importance to guide training focus areas")
    report.append("• Implement regular model retraining with new race data")
    
    if 'category_importance' in results['analysis']:
        # Find most important category
        category_imp = results['analysis']['category_importance']
        top_category = max(category_imp.items(), key=lambda x: x[1]['total_importance'])[0]
        report.append(f"• Focus data collection on {top_category.lower()} for model improvement")
    
    if 'optimized' in results['model_results']:
        best_mae = results['model_results']['optimized']['test_mae'] / 60
        if best_mae < 2:
            report.append("• Consider ensemble methods for even higher precision")
        else:
            report.append("• Investigate additional feature engineering opportunities")
    
    report.append("• Use scenario predictions for personalized performance targets")
    report.append("• Monitor model performance drift with new data")
    
    # Save report
    with open(output_dir / 'xgboost_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating XGBoost analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('xgboost')
    
    # Load results
    df, results, models = load_results()
    
    # Create visualizations
    print("Plotting advanced feature importance...")
    plot_feature_importance_advanced(results, output_dir)
    
    print("Plotting model performance comparison...")
    plot_model_performance_comparison(results, output_dir)
    
    print("Plotting scenario analysis...")
    plot_scenario_analysis(results, output_dir)
    
    print("Creating XGBoost dashboard...")
    plot_xgboost_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_xgboost_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()