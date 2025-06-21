#!/usr/bin/env python3
"""
Survival Analysis Visualization
Creates comprehensive visualizations for survival analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
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
    """Load survival analysis results and data."""
    # Load processed data
    df = pd.read_csv('data/processed/survival_analysis_data.csv')
    
    # Load results
    with open('data/processed/survival_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    # Load model objects
    try:
        with open('data/processed/survival_analysis_models.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}
    
    return df, results, models

def plot_kaplan_meier_curves(df, results, output_dir):
    """Plot Kaplan-Meier survival curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Survival curves by gender
    for gender in ['MALE', 'FEMALE']:
        if gender in results['kaplan_meier']:
            gender_data = df[df['gender'] == gender]
            if len(gender_data) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(gender_data['duration'], 
                       event_observed=gender_data['event'],
                       label=f'{gender} (n={len(gender_data)})')
                kmf.plot_survival_function(ax=ax1)
    
    ax1.set_title('Survival Curves by Gender', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Survival Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Survival curves by age group
    if 'age_group' in df.columns:
        for age_group in df['age_group'].dropna().unique():
            age_data = df[df['age_group'] == age_group]
            if len(age_data) > 10:  # Only plot if sufficient data
                kmf = KaplanMeierFitter()
                kmf.fit(age_data['duration'], 
                       event_observed=age_data['event'],
                       label=f'{age_group} (n={len(age_data)})')
                kmf.plot_survival_function(ax=ax2)
    
    ax2.set_title('Survival Curves by Age Group', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Survival Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kaplan_meier_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cox_model_results(results, models, output_dir):
    """Plot Cox proportional hazards model results."""
    if 'cox_model' not in results or 'cox_model' not in models:
        print("Cox model not available for plotting")
        return
    
    cox_model = models['cox_model']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Hazard ratios with confidence intervals
    hazard_ratios = pd.Series(results['cox_model']['hazard_ratios'])
    ci_lower = pd.DataFrame(results['cox_model']['confidence_intervals'])['lower 95%']
    ci_upper = pd.DataFrame(results['cox_model']['confidence_intervals'])['upper 95%']
    
    y_pos = np.arange(len(hazard_ratios))
    ax1.errorbar(hazard_ratios.values, y_pos, 
                xerr=[hazard_ratios.values - ci_lower.values, 
                      ci_upper.values - hazard_ratios.values],
                fmt='o', capsize=5, capthick=2)
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='HR = 1')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(hazard_ratios.index, fontsize=10)
    ax1.set_xlabel('Hazard Ratio')
    ax1.set_title('Cox Model Hazard Ratios with 95% CI', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    try:
        residuals = cox_model.compute_residuals(df_cox, kind='martingale')
        ax2.scatter(range(len(residuals)), residuals, alpha=0.6)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel('Observation Index')
        ax2.set_ylabel('Martingale Residuals')
        ax2.set_title('Cox Model Residuals', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'Residuals not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Cox Model Residuals (Not Available)', fontweight='bold')
    
    # Plot 3: Partial hazard plot for age
    if any('age_' in col for col in hazard_ratios.index):
        age_coeffs = hazard_ratios[[col for col in hazard_ratios.index if 'age_' in col]]
        age_groups = [col.replace('age_', '') for col in age_coeffs.index]
        
        ax3.bar(age_groups, age_coeffs.values)
        ax3.set_xlabel('Age Group')
        ax3.set_ylabel('Log Hazard Ratio')
        ax3.set_title('Age Effect on Hazard', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Age data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Summary statistics
    ax4.axis('off')
    summary_text = f"""
    Cox Proportional Hazards Model Summary
    
    Concordance Index: {results['cox_model']['concordance_index']:.3f}
    Log-likelihood: {results['cox_model']['log_likelihood']:.2f}
    AIC: {results['cox_model']['AIC']:.2f}
    
    Interpretation:
    • HR > 1: Increased hazard (slower performance)
    • HR < 1: Decreased hazard (faster performance)
    • HR = 1: No effect
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cox_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_distributions(df, results, output_dir):
    """Plot performance distribution analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Duration distribution by gender
    if 'gender' in df.columns:
        for gender in df['gender'].unique():
            if pd.notna(gender):
                gender_data = df[df['gender'] == gender]['duration']
                ax1.hist(gender_data/60, bins=30, alpha=0.7, label=f'{gender}', density=True)
        
        ax1.set_xlabel('Completion Time (minutes)')
        ax1.set_ylabel('Density')
        ax1.set_title('Completion Time Distribution by Gender', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Age vs Performance
    if 'age' in df.columns:
        ax2.scatter(df['age'], df['duration']/60, alpha=0.6, c=df['gender'].map({'MALE': 'blue', 'FEMALE': 'red'}))
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Completion Time (minutes)')
        ax2.set_title('Age vs Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance quartiles
    if 'quartile_analysis' in results:
        quartiles = results['quartile_analysis']['quartiles']
        quartile_times = [v/60 for v in quartiles.values()]
        quartile_labels = ['25th', '50th (Median)', '75th']
        
        ax3.bar(quartile_labels, quartile_times, color=['green', 'orange', 'red'])
        ax3.set_ylabel('Time (minutes)')
        ax3.set_title('Performance Quartiles', fontweight='bold')
        for i, v in enumerate(quartile_times):
            ax3.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Club membership effect
    if 'has_club' in df.columns:
        club_data = df.groupby('has_club')['duration'].apply(lambda x: x/60)
        club_data.plot(kind='box', ax=ax4)
        ax4.set_xticklabels(['No Club', 'Has Club'])
        ax4.set_ylabel('Completion Time (minutes)')
        ax4.set_title('Performance by Club Membership', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_dashboard(df, results, output_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main KM curve
    ax1 = fig.add_subplot(gs[0, :2])
    for gender in ['MALE', 'FEMALE']:
        if gender in df['gender'].values:
            gender_data = df[df['gender'] == gender]
            if len(gender_data) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(gender_data['duration'], 
                       event_observed=gender_data['event'],
                       label=f'{gender}')
                kmf.plot_survival_function(ax=ax1)
    ax1.set_title('Kaplan-Meier Survival Curves', fontsize=16, fontweight='bold')
    ax1.legend()
    
    # Performance histogram
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.hist(df['duration']/60, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Completion Time (minutes)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Performance Distribution', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Age distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if 'age' in df.columns:
        df['age'].hist(bins=20, ax=ax3, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Count')
        ax3.set_title('Age Distribution', fontweight='bold')
    
    # Gender distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        ax4.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
        ax4.set_title('Gender Distribution', fontweight='bold')
    
    # Club membership
    ax5 = fig.add_subplot(gs[1, 2])
    if 'has_club' in df.columns:
        club_counts = df['has_club'].map({0: 'No Club', 1: 'Has Club'}).value_counts()
        ax5.pie(club_counts.values, labels=club_counts.index, autopct='%1.1f%%')
        ax5.set_title('Club Membership', fontweight='bold')
    
    # Key statistics
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    stats_text = f"""
    KEY STATISTICS
    
    Total Participants: {len(df):,}
    Completion Rate: {df['event'].mean():.1%}
    
    Median Time: {df['duration'].median()/60:.1f} min
    Mean Time: {df['duration'].mean()/60:.1f} min
    Std Dev: {df['duration'].std()/60:.1f} min
    
    Age Range: {df['age'].min():.0f} - {df['age'].max():.0f}
    Mean Age: {df['age'].mean():.1f}
    """
    
    if 'logrank_test' in results:
        stats_text += f"\nGender Diff p-value: {results['logrank_test']['p_value']:.3f}"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # Cox model results if available
    ax7 = fig.add_subplot(gs[2, :])
    if 'cox_model' in results:
        hazard_ratios = pd.Series(results['cox_model']['hazard_ratios'])
        y_pos = np.arange(len(hazard_ratios))
        ax7.barh(y_pos, hazard_ratios.values)
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(hazard_ratios.index, fontsize=10)
        ax7.axvline(x=1, color='red', linestyle='--', alpha=0.7)
        ax7.set_xlabel('Hazard Ratio')
        ax7.set_title('Cox Model Hazard Ratios (HR > 1 = Higher Risk of Slower Performance)', fontweight='bold')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Cox Model Results Not Available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=16)
    
    plt.suptitle('Race Performance Survival Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'survival_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_survival_report(df, results, output_dir):
    """Generate a text report summarizing survival analysis findings."""
    report = []
    report.append("SURVIVAL ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants\n")
    
    # Basic statistics
    report.append("BASIC STATISTICS")
    report.append("-" * 20)
    report.append(f"Completion Rate: {df['event'].mean():.1%}")
    report.append(f"Median Completion Time: {df['duration'].median()/60:.1f} minutes")
    report.append(f"Mean Completion Time: {df['duration'].mean()/60:.1f} minutes")
    report.append(f"Standard Deviation: {df['duration'].std()/60:.1f} minutes")
    report.append(f"Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    report.append(f"Mean Age: {df['age'].mean():.1f} years\n")
    
    # Gender analysis
    if 'logrank_test' in results:
        report.append("GENDER COMPARISON")
        report.append("-" * 20)
        report.append(f"Logrank test p-value: {results['logrank_test']['p_value']:.4f}")
        if results['logrank_test']['p_value'] < 0.05:
            report.append("Significant difference in performance between genders")
        else:
            report.append("No significant difference in performance between genders")
        report.append("")
    
    # Cox model results
    if 'cox_model' in results:
        report.append("COX PROPORTIONAL HAZARDS MODEL")
        report.append("-" * 35)
        report.append(f"Concordance Index: {results['cox_model']['concordance_index']:.3f}")
        report.append(f"Model AIC: {results['cox_model']['AIC']:.2f}")
        report.append("\nHazard Ratios (>1 indicates higher risk of slower performance):")
        
        hazard_ratios = results['cox_model']['hazard_ratios']
        for factor, hr in hazard_ratios.items():
            report.append(f"  {factor}: {hr:.3f}")
        report.append("")
    
    # Performance quartiles
    if 'quartile_analysis' in results:
        report.append("PERFORMANCE QUARTILES")
        report.append("-" * 25)
        quartiles = results['quartile_analysis']['quartiles']
        for q, time in quartiles.items():
            report.append(f"{q*100:.0f}th percentile: {time/60:.1f} minutes")
        report.append("")
    
    # Recommendations
    report.append("KEY FINDINGS & RECOMMENDATIONS")
    report.append("-" * 35)
    
    # Generate findings based on results
    if 'cox_model' in results:
        hazard_ratios = results['cox_model']['hazard_ratios']
        
        # Age effects
        age_effects = {k: v for k, v in hazard_ratios.items() if 'age_' in k}
        if age_effects:
            max_age_hr = max(age_effects.items(), key=lambda x: x[1])
            report.append(f"• Age group with highest performance risk: {max_age_hr[0]} (HR: {max_age_hr[1]:.2f})")
        
        # Club effects
        if 'has_club' in hazard_ratios:
            club_hr = hazard_ratios['has_club']
            if club_hr < 1:
                report.append(f"• Club membership appears beneficial (HR: {club_hr:.2f})")
            else:
                report.append(f"• Club membership shows higher risk (HR: {club_hr:.2f})")
    
    completion_rate = df['event'].mean()
    if completion_rate < 0.95:
        report.append(f"• Consider investigating factors leading to {(1-completion_rate)*100:.1f}% non-completion rate")
    
    report.append("• Use survival curves to set realistic time targets for different demographic groups")
    report.append("• Consider age-specific training programs based on hazard ratio analysis")
    
    # Save report
    with open(output_dir / 'survival_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating survival analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('survival_analysis')
    
    # Load results
    df, results, models = load_results()
    
    # Create visualizations
    print("Plotting Kaplan-Meier curves...")
    plot_kaplan_meier_curves(df, results, output_dir)
    
    print("Plotting Cox model results...")
    plot_cox_model_results(results, models, output_dir)
    
    print("Plotting performance distributions...")
    plot_performance_distributions(df, results, output_dir)
    
    print("Creating summary dashboard...")
    plot_summary_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_survival_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()