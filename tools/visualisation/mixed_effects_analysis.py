#!/usr/bin/env python3
"""
Mixed Effects Models Visualization
Creates comprehensive visualizations for mixed effects analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
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
    """Load mixed effects analysis results and data."""
    # Load processed data
    df = pd.read_csv('data/processed/mixed_effects_data.csv')
    
    # Load results
    with open('data/processed/mixed_effects_results.json', 'r') as f:
        results = json.load(f)
    
    # Load model objects
    try:
        with open('data/processed/mixed_effects_models.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}
    
    return df, results, models

def plot_hierarchical_effects(df, results, output_dir):
    """Plot hierarchical effects: club and age group performance distributions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Club performance distribution
    if 'club_performance' in results['group_analysis']:
        club_data = pd.DataFrame(results['group_analysis']['club_performance']).T
        club_data = club_data[club_data['count'] >= 5].sort_values('mean_time')
        
        top_clubs = club_data.head(15)  # Top 15 clubs by performance
        
        ax1.barh(range(len(top_clubs)), top_clubs['mean_time']/60, 
                xerr=top_clubs['std_time']/60, capsize=3)
        ax1.set_yticks(range(len(top_clubs)))
        ax1.set_yticklabels(top_clubs.index, fontsize=8)
        ax1.set_xlabel('Mean Finish Time (minutes)')
        ax1.set_title('Club Performance Comparison\n(Top 15 by Performance)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Age group performance
    if 'age_group_performance' in results['group_analysis']:
        age_data = pd.DataFrame(results['group_analysis']['age_group_performance']).T
        
        ax2.bar(range(len(age_data)), age_data['mean_time']/60, 
               yerr=age_data['std_time']/60, capsize=5, alpha=0.7)
        ax2.set_xticks(range(len(age_data)))
        ax2.set_xticklabels(age_data.index, rotation=45)
        ax2.set_ylabel('Mean Finish Time (minutes)')
        ax2.set_title('Performance by Age Group', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance variability by club size
    if 'club_performance' in results['group_analysis']:
        club_data = pd.DataFrame(results['group_analysis']['club_performance']).T
        club_data = club_data[club_data['count'] >= 5]
        
        ax3.scatter(club_data['count'], club_data['cv_time'], alpha=0.6, s=60)
        ax3.set_xlabel('Club Size (number of members)')
        ax3.set_ylabel('Coefficient of Variation (time)')
        ax3.set_title('Performance Consistency vs Club Size', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(club_data) > 3:
            z = np.polyfit(club_data['count'], club_data['cv_time'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(club_data['count'].min(), club_data['count'].max(), 100)
            ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    # Plot 4: Gender performance by age
    if 'gender_age_performance' in results['group_analysis']:
        gender_age_data = pd.DataFrame(results['group_analysis']['gender_age_performance']).T
        
        # Reshape data for plotting
        plot_data = []
        for idx, row in gender_age_data.iterrows():
            gender, age_group = idx
            plot_data.append({
                'gender': gender,
                'age_group': age_group,
                'mean_time': row['mean']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(plot_df) > 0:
            pivot_data = plot_df.pivot(index='age_group', columns='gender', values='mean_time')
            pivot_data = pivot_data / 60  # Convert to minutes
            
            x = np.arange(len(pivot_data.index))
            width = 0.35
            
            if 'MALE' in pivot_data.columns:
                ax4.bar(x - width/2, pivot_data['MALE'], width, label='Male', alpha=0.7)
            if 'FEMALE' in pivot_data.columns:
                ax4.bar(x + width/2, pivot_data['FEMALE'], width, label='Female', alpha=0.7)
            
            ax4.set_xlabel('Age Group')
            ax4.set_ylabel('Mean Finish Time (minutes)')
            ax4.set_title('Gender Performance by Age Group', fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(pivot_data.index, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hierarchical_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results, output_dir):
    """Plot model comparison and diagnostics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model comparison metrics
    if 'model_comparison' in results['model_results']:
        model_comp = results['model_results']['model_comparison']
        
        models = list(model_comp.keys())
        aic_values = [model_comp[m]['AIC'] for m in models]
        bic_values = [model_comp[m]['BIC'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, aic_values, width, label='AIC', alpha=0.7)
        ax1.bar(x + width/2, bic_values, width, label='BIC', alpha=0.7)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Information Criterion')
        ax1.set_title('Model Comparison (Lower is Better)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fixed effects coefficients
    fixed_effects_plotted = False
    for model_name, model_result in results['model_results'].items():
        if 'fixed_effects' in model_result and 'error' not in model_result:
            fixed_effects = model_result['fixed_effects']
            p_values = model_result.get('fixed_effects_pvalues', {})
            
            effects = list(fixed_effects.keys())
            coeffs = list(fixed_effects.values())
            
            # Color bars by significance
            colors = ['red' if p_values.get(e, 1) < 0.05 else 'gray' for e in effects]
            
            ax2.barh(range(len(effects)), coeffs, color=colors, alpha=0.7)
            ax2.set_yticks(range(len(effects)))
            ax2.set_yticklabels(effects, fontsize=10)
            ax2.set_xlabel('Coefficient Value')
            ax2.set_title(f'Fixed Effects: {model_name.replace("_", " ").title()}', fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            # Add significance legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='p < 0.05'),
                             Patch(facecolor='gray', label='p ≥ 0.05')]
            ax2.legend(handles=legend_elements, loc='lower right')
            
            fixed_effects_plotted = True
            break
    
    if not fixed_effects_plotted:
        ax2.text(0.5, 0.5, 'Fixed effects not available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Plot 3: Random effects (if available)
    if 'random_effects' in results['model_results']:
        random_effects = results['model_results']['random_effects']
        
        # Plot random effects for the first available model
        re_plotted = False
        for model_name, re_dict in random_effects.items():
            if re_dict:
                groups = list(re_dict.keys())[:15]  # Top 15 groups
                effects = [re_dict[g] for g in groups]
                
                ax3.barh(range(len(groups)), effects, alpha=0.7)
                ax3.set_yticks(range(len(groups)))
                ax3.set_yticklabels(groups, fontsize=8)
                ax3.set_xlabel('Random Effect')
                ax3.set_title(f'Random Effects: {model_name.replace("_", " ").title()}', fontweight='bold')
                ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                ax3.grid(True, alpha=0.3)
                re_plotted = True
                break
        
        if not re_plotted:
            ax3.text(0.5, 0.5, 'Random effects not available', 
                    ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Variance components
    variance_data = []
    for model_name, model_result in results['model_results'].items():
        if 'error' not in model_result:
            random_var = model_result.get('random_effects_var')
            residual_var = model_result.get('residual_var')
            
            if random_var is not None and residual_var is not None:
                total_var = random_var + residual_var
                variance_data.append({
                    'model': model_name.replace('_', ' ').title(),
                    'random': random_var / total_var * 100,
                    'residual': residual_var / total_var * 100
                })
    
    if variance_data:
        var_df = pd.DataFrame(variance_data)
        
        x = np.arange(len(var_df))
        ax4.bar(x, var_df['random'], label='Random Effects', alpha=0.7)
        ax4.bar(x, var_df['residual'], bottom=var_df['random'], label='Residual', alpha=0.7)
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Variance Explained (%)')
        ax4.set_title('Variance Decomposition', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(var_df['model'], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Variance data not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_predictions(df, results, output_dir):
    """Plot performance predictions and residuals."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Actual vs Predicted performance by age
    ax1.scatter(df['age'], df['finish_time']/60, alpha=0.4, s=20, label='Actual')
    
    # Add trend line
    age_range = np.linspace(df['age'].min(), df['age'].max(), 100)
    # Simple polynomial fit for visualization
    z = np.polyfit(df['age'].dropna(), df[df['age'].notna()]['finish_time']/60, 2)
    p = np.poly1d(z)
    ax1.plot(age_range, p(age_range), 'r-', linewidth=2, label='Trend')
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Finish Time (minutes)')
    ax1.set_title('Performance vs Age', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance by club membership
    club_comparison = df.groupby('has_club')['finish_time'].apply(lambda x: x/60)
    box_data = [club_comparison[0], club_comparison[1]]
    labels = ['No Club', 'Has Club']
    
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Finish Time (minutes)')
    ax2.set_title('Performance by Club Membership', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log-transformed performance distribution
    ax3.hist(df['log_finish_time'], bins=50, alpha=0.7, density=True, color='skyblue')
    ax3.set_xlabel('Log(Finish Time)')
    ax3.set_ylabel('Density')
    ax3.set_title('Log-Transformed Performance Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Overlay normal distribution for comparison
    mu, sigma = df['log_finish_time'].mean(), df['log_finish_time'].std()
    x = np.linspace(df['log_finish_time'].min(), df['log_finish_time'].max(), 100)
    normal_curve = ((1/np.sqrt(2*np.pi*sigma**2)) * 
                   np.exp(-0.5*((x-mu)/sigma)**2))
    ax3.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
    ax3.legend()
    
    # Plot 4: Performance predictions scenarios
    if 'performance_predictions' in results['group_analysis']:
        pred_data = results['group_analysis']['performance_predictions']
        
        scenarios = []
        times = []
        
        for scenario, pred in pred_data.items():
            if pred is not None:
                scenarios.append(scenario.replace('_', ' '))
                times.append(pred['time_minutes'])
        
        if scenarios:
            colors = ['blue' if 'Male' in s else 'red' for s in scenarios]
            bars = ax4.bar(range(len(scenarios)), times, color=colors, alpha=0.7)
            
            ax4.set_xlabel('Scenario')
            ax4.set_ylabel('Predicted Time (minutes)')
            ax4.set_title('Performance Predictions by Scenario', fontweight='bold')
            ax4.set_xticks(range(len(scenarios)))
            ax4.set_xticklabels(scenarios, rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Prediction data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_mixed_effects_dashboard(df, results, output_dir):
    """Create a comprehensive dashboard for mixed effects analysis."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main performance distribution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(df['finish_time']/60, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Finish Time (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Performance Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Age distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['age'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Count')
    ax2.set_title('Age Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Gender distribution
    ax3 = fig.add_subplot(gs[0, 3])
    gender_counts = df['gender'].value_counts()
    ax3.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    ax3.set_title('Gender Distribution', fontweight='bold')
    
    # Club performance comparison (top 10)
    ax4 = fig.add_subplot(gs[1, :2])
    if 'club_performance' in results['group_analysis']:
        club_data = pd.DataFrame(results['group_analysis']['club_performance']).T
        club_data = club_data[club_data['count'] >= 5].sort_values('mean_time').head(10)
        
        ax4.barh(range(len(club_data)), club_data['mean_time']/60, alpha=0.7)
        ax4.set_yticks(range(len(club_data)))
        ax4.set_yticklabels(club_data.index, fontsize=9)
        ax4.set_xlabel('Mean Finish Time (minutes)')
        ax4.set_title('Top 10 Club Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # Model comparison
    ax5 = fig.add_subplot(gs[1, 2:])
    if 'model_comparison' in results['model_results']:
        model_comp = results['model_results']['model_comparison']
        models = list(model_comp.keys())
        aic_values = [model_comp[m]['AIC'] for m in models]
        
        ax5.bar(range(len(models)), aic_values, alpha=0.7, color='orange')
        ax5.set_xlabel('Model')
        ax5.set_ylabel('AIC (lower is better)')
        ax5.set_title('Model Comparison', fontweight='bold')
        ax5.set_xticks(range(len(models)))
        ax5.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
        ax5.grid(True, alpha=0.3)
    
    # Key statistics
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Calculate key statistics
    stats_text = f"""
    MIXED EFFECTS MODEL ANALYSIS SUMMARY
    
    Dataset Overview:
    • Total Participants: {len(df):,}
    • Mean Finish Time: {df['finish_time'].mean()/60:.1f} minutes
    • Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years
    • Clubs with 5+ members: {len([k for k, v in results['group_analysis']['club_performance']['count'].items() if v >= 5])}
    
    Model Performance:
    """
    
    # Add ICC if available
    if 'icc_club' in results['group_analysis'] and results['group_analysis']['icc_club']:
        icc_value = results['group_analysis']['icc_club']
        stats_text += f"• Intraclass Correlation (Club): {icc_value:.3f}\n"
        if icc_value > 0.1:
            stats_text += "  → Strong club effect on performance\n"
        elif icc_value > 0.05:
            stats_text += "  → Moderate club effect on performance\n"
        else:
            stats_text += "  → Weak club effect on performance\n"
    
    # Add best model info
    if 'model_comparison' in results['model_results']:
        best_model = min(results['model_results']['model_comparison'].items(), 
                        key=lambda x: x[1]['AIC'])[0]
        stats_text += f"• Best Model: {best_model.replace('_', ' ').title()}\n"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Mixed Effects Model Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'mixed_effects_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_mixed_effects_report(df, results, output_dir):
    """Generate a comprehensive text report."""
    report = []
    report.append("MIXED EFFECTS MODEL ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants\n")
    
    # Data summary
    report.append("DATA SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Participants: {len(df):,}")
    report.append(f"Mean Finish Time: {df['finish_time'].mean()/60:.1f} minutes")
    report.append(f"Median Finish Time: {df['finish_time'].median()/60:.1f} minutes")
    report.append(f"Standard Deviation: {df['finish_time'].std()/60:.1f} minutes")
    report.append(f"Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    report.append(f"Gender Distribution: {df['gender'].value_counts().to_dict()}")
    report.append("")
    
    # Model comparison
    if 'model_comparison' in results['model_results']:
        report.append("MODEL COMPARISON")
        report.append("-" * 20)
        model_comp = results['model_results']['model_comparison']
        
        for model, metrics in model_comp.items():
            report.append(f"{model.replace('_', ' ').title()}:")
            report.append(f"  AIC: {metrics['AIC']:.2f}")
            report.append(f"  BIC: {metrics['BIC']:.2f}")
            report.append(f"  Log-Likelihood: {metrics['Log-Likelihood']:.2f}")
        
        best_model = min(model_comp.items(), key=lambda x: x[1]['AIC'])[0]
        report.append(f"\nBest Model (lowest AIC): {best_model.replace('_', ' ').title()}")
        report.append("")
    
    # Fixed effects
    report.append("FIXED EFFECTS ANALYSIS")
    report.append("-" * 25)
    
    for model_name, model_result in results['model_results'].items():
        if 'fixed_effects' in model_result and 'error' not in model_result:
            report.append(f"\n{model_name.replace('_', ' ').title()} Model:")
            fixed_effects = model_result['fixed_effects']
            p_values = model_result.get('fixed_effects_pvalues', {})
            
            for effect, coeff in fixed_effects.items():
                p_val = p_values.get(effect, 'N/A')
                significance = " ***" if p_val != 'N/A' and p_val < 0.001 else \
                             " **" if p_val != 'N/A' and p_val < 0.01 else \
                             " *" if p_val != 'N/A' and p_val < 0.05 else ""
                report.append(f"  {effect}: {coeff:.4f} (p={p_val}){significance}")
    
    report.append("")
    
    # ICC analysis
    if 'icc_club' in results['group_analysis'] and results['group_analysis']['icc_club']:
        report.append("INTRACLASS CORRELATION ANALYSIS")
        report.append("-" * 35)
        icc_value = results['group_analysis']['icc_club']
        report.append(f"Club ICC: {icc_value:.3f}")
        
        if icc_value > 0.1:
            interpretation = "Strong clustering effect - club membership significantly affects performance"
        elif icc_value > 0.05:
            interpretation = "Moderate clustering effect - club membership has noticeable impact"
        else:
            interpretation = "Weak clustering effect - minimal impact of club membership"
        
        report.append(f"Interpretation: {interpretation}")
        report.append("")
    
    # Group analysis
    report.append("GROUP PERFORMANCE ANALYSIS")
    report.append("-" * 30)
    
    if 'club_performance' in results['group_analysis']:
        club_data = pd.DataFrame(results['group_analysis']['club_performance']).T
        top_clubs = club_data[club_data['count'] >= 5].sort_values('mean_time').head(5)
        
        report.append("Top 5 Performing Clubs (5+ members):")
        for idx, (club, data) in enumerate(top_clubs.iterrows(), 1):
            report.append(f"  {idx}. {club}: {data['mean_time']/60:.1f} min (n={data['count']})")
        report.append("")
    
    if 'age_group_performance' in results['group_analysis']:
        age_data = pd.DataFrame(results['group_analysis']['age_group_performance']).T
        report.append("Performance by Age Group:")
        for age_group, data in age_data.iterrows():
            report.append(f"  {age_group}: {data['mean']/60:.1f} min ± {data['std']/60:.1f} (n={data['count']})")
        report.append("")
    
    # Recommendations
    report.append("KEY FINDINGS & RECOMMENDATIONS")
    report.append("-" * 35)
    
    # Generate specific findings
    if 'icc_club' in results['group_analysis'] and results['group_analysis']['icc_club']:
        icc_value = results['group_analysis']['icc_club']
        if icc_value > 0.1:
            report.append("• Strong club effects detected - consider club-specific training programs")
        elif icc_value > 0.05:
            report.append("• Moderate club effects - club culture influences performance")
    
    if 'club_performance' in results['group_analysis']:
        club_data = pd.DataFrame(results['group_analysis']['club_performance']).T
        club_data = club_data[club_data['count'] >= 5]
        cv_range = club_data['cv_time'].max() - club_data['cv_time'].min()
        
        if cv_range > 0.1:
            report.append("• Large variation in club consistency - some clubs show much more uniform performance")
        
        best_club_size = club_data.loc[club_data['mean_time'].idxmin(), 'count']
        report.append(f"• Best performing club has {best_club_size} members - consider optimal club size analysis")
    
    report.append("• Use hierarchical modeling for performance prediction across different groups")
    report.append("• Consider age-specific and club-specific interventions")
    
    # Save report
    with open(output_dir / 'mixed_effects_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating mixed effects analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('mixed_effects')
    
    # Load results
    df, results, models = load_results()
    
    # Create visualizations
    print("Plotting hierarchical effects...")
    plot_hierarchical_effects(df, results, output_dir)
    
    print("Plotting model comparison...")
    plot_model_comparison(results, output_dir)
    
    print("Plotting performance predictions...")
    plot_performance_predictions(df, results, output_dir)
    
    print("Creating summary dashboard...")
    plot_mixed_effects_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_mixed_effects_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()