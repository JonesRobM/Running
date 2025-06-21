#!/usr/bin/env python3
"""
Time Series Analysis Visualization
Creates comprehensive visualizations for time series analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
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
    """Load time series analysis results and data."""
    # Load processed data
    df = pd.read_csv('data/processed/time_series_data.csv')
    
    # Load results
    with open('data/processed/time_series_results.json', 'r') as f:
        results = json.load(f)
    
    return df, results

def plot_pacing_analysis(df, results, output_dir):
    """Plot comprehensive pacing strategy analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Pacing strategy distribution
    if 'pacing_analysis' in results and 'error' not in results['pacing_analysis']:
        pacing_dist = results['pacing_analysis']['pacing_distribution']
        
        strategies = list(pacing_dist.keys())
        counts = list(pacing_dist.values())
        
        colors = ['green', 'blue', 'orange', 'red'][:len(strategies)]
        bars = ax1.bar(strategies, counts, color=colors, alpha=0.7)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count}\n({count/total*100:.1f}%)', 
                    ha='center', va='bottom', fontweight='bold')
        
        ax1.set_xlabel('Pacing Strategy')
        ax1.set_ylabel('Number of Runners')
        ax1.set_title('Distribution of Pacing Strategies', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Pacing data not available', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: Performance by pacing strategy
    if 'pacing_analysis' in results and 'performance_by_pacing' in results['pacing_analysis']:
        perf_by_pacing = results['pacing_analysis']['performance_by_pacing']
        
        strategies = list(perf_by_pacing.keys())
        mean_times = [perf_by_pacing[s]['mean_time']/60 for s in strategies]
        
        bars = ax2.bar(strategies, mean_times, alpha=0.7, color='steelblue')
        
        # Add time labels
        for bar, time in zip(bars, mean_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Pacing Strategy')
        ax2.set_ylabel('Mean Finish Time (minutes)')
        ax2.set_title('Performance by Pacing Strategy', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pace ratio distribution
    if 'pace_ratio' in df.columns:
        pace_ratios = df['pace_ratio'].dropna()
        
        ax3.hist(pace_ratios, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Even Pace')
        ax3.axvline(x=pace_ratios.mean(), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean ({pace_ratios.mean():.2f})')
        
        ax3.set_xlabel('Pace Ratio (Second Half / First Half)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Pace Ratios', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Pace ratio data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Age vs pacing strategy
    if 'demographic_pacing' in results['pacing_analysis'] and 'age_group' in results['pacing_analysis']['demographic_pacing']:
        age_pacing = results['pacing_analysis']['demographic_pacing']['age_group']
        
        age_groups = list(age_pacing['pace_ratio']['mean'].keys())
        pace_ratios = list(age_pacing['pace_ratio']['mean'].values())
        negative_split_rates = list(age_pacing['negative_split']['mean'].values())
        
        x = np.arange(len(age_groups))
        width = 0.35
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x - width/2, pace_ratios, width, label='Mean Pace Ratio', alpha=0.7)
        bars2 = ax4_twin.bar(x + width/2, [r*100 for r in negative_split_rates], width, 
                           label='Negative Split %', alpha=0.7, color='orange')
        
        ax4.set_xlabel('Age Group')
        ax4.set_ylabel('Mean Pace Ratio', color='blue')
        ax4_twin.set_ylabel('Negative Split Rate (%)', color='orange')
        ax4.set_title('Pacing Strategy by Age Group', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(age_groups)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pacing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_trends(df, results, output_dir):
    """Plot performance trends and time series patterns."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Performance distribution with trend line
    sorted_df = df.sort_values('finish_time').reset_index(drop=True)
    sorted_df['position'] = range(1, len(sorted_df) + 1)
    
    ax1.scatter(sorted_df['position'], sorted_df['finish_time']/60, alpha=0.6, s=20)
    
    # Add trend line
    if 'trend_analysis' in results['time_series_decomposition']:
        trend_analysis = results['time_series_decomposition']['trend_analysis']
        best_order = max(trend_analysis.keys(), key=lambda x: trend_analysis[x]['r_squared'])
        coeffs = trend_analysis[best_order]['coefficients']
        
        x_trend = np.linspace(1, len(sorted_df), 100)
        y_trend = np.polyval(coeffs, x_trend)
        ax1.plot(x_trend, y_trend/60, 'r-', linewidth=2, 
                label=f'Trend (R² = {trend_analysis[best_order]["r_squared"]:.3f})')
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Finish Time (minutes)')
    ax1.set_title('Performance Progression Through Field', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time gaps between consecutive finishers
    if 'gap_analysis' in results['performance_trends']:
        gap_analysis = results['performance_trends']['gap_analysis']
        
        time_gaps = np.diff(sorted_df['finish_time'].values)
        
        ax2.plot(range(1, len(time_gaps) + 1), time_gaps, alpha=0.7, linewidth=1)
        ax2.axhline(y=gap_analysis['mean_gap'], color='red', linestyle='--', 
                   label=f'Mean Gap ({gap_analysis["mean_gap"]:.1f}s)')
        ax2.axhline(y=gap_analysis['mean_gap'] + 2*gap_analysis['std_gap'], 
                   color='orange', linestyle='--', alpha=0.7,
                   label='Mean + 2σ')
        
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Time Gap to Next Runner (seconds)')
        ax2.set_title('Time Gaps Between Consecutive Finishers', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Age-performance relationship
    if 'age_performance_trend' in results['performance_trends']:
        age_trend = results['performance_trends']['age_performance_trend']
        
        ages = age_trend['ages']
        mean_times = [t/60 for t in age_trend['mean_times']]
        counts = age_trend['participant_counts']
        
        # Scatter plot sized by participant count
        scatter = ax3.scatter(ages, mean_times, s=[c*3 for c in counts], alpha=0.6, c=counts, cmap='viridis')
        
        # Add polynomial trend if available
        if 'polynomial_coeffs' in age_trend:
            coeffs = age_trend['polynomial_coeffs']
            age_range = np.linspace(min(ages), max(ages), 100)
            time_trend = np.polyval(coeffs, age_range)
            ax3.plot(age_range, time_trend/60, 'r-', linewidth=2, label='Quadratic Trend')
            
            # Mark optimal age if available
            if 'optimal_age' in age_trend:
                optimal_age = age_trend['optimal_age']
                optimal_time = np.polyval(coeffs, optimal_age)
                ax3.plot(optimal_age, optimal_time/60, 'ro', markersize=10, 
                        label=f'Optimal Age ({optimal_age:.1f})')
        
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Mean Finish Time (minutes)')
        ax3.set_title('Age-Performance Relationship', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for participant counts
        plt.colorbar(scatter, ax=ax3, label='Participant Count')
    
    # Plot 4: Performance quartiles analysis
    if 'performance_quartiles' in results['performance_trends']:
        quartiles = results['performance_trends']['performance_quartiles']
        
        quartile_names = list(quartiles.keys())
        quartile_times = [quartiles[q]/60 for q in quartile_names]
        
        bars = ax4.bar(quartile_names, quartile_times, alpha=0.7, color='lightgreen')
        
        # Add time labels
        for bar, time in zip(bars, quartile_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_xlabel('Percentile')
        ax4.set_ylabel('Finish Time (minutes)')
        ax4.set_title('Performance Quartiles', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_outlier_analysis(df, results, output_dir):
    """Plot outlier detection and anomaly analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Box plot with outliers
    times = df['finish_time'].dropna() / 60
    
    bp = ax1.boxplot(times, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Mark outliers if available
    if 'outlier_analysis' in results and 'outlier_analysis' in results['outlier_analysis']:
        outlier_info = results['outlier_analysis']['outlier_analysis']
        iqr_outliers = outlier_info['iqr_outliers']
        
        if iqr_outliers['count'] > 0:
            ax1.axhline(y=iqr_outliers['lower_bound']/60, color='red', linestyle='--', 
                       label=f'IQR Bounds')
            ax1.axhline(y=iqr_outliers['upper_bound']/60, color='red', linestyle='--')
    
    ax1.set_ylabel('Finish Time (minutes)')
    ax1.set_title('Performance Distribution with Outliers', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Z-score distribution
    z_scores = np.abs(stats.zscore(times))
    
    ax2.hist(z_scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Z-score = 3')
    ax2.axvline(x=2, color='orange', linestyle='--', linewidth=2, label='Z-score = 2')
    
    ax2.set_xlabel('Absolute Z-score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Z-score Distribution (Outlier Detection)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pacing anomalies (if available)
    if 'pacing_anomalies' in results['outlier_analysis']:
        pacing_anom = results['outlier_analysis']['pacing_anomalies']
        
        categories = ['Extreme\nNegative', 'Normal\nPacing', 'Extreme\nPositive']
        counts = [
            pacing_anom['extreme_negative_splits'],
            df['pace_ratio'].notna().sum() - pacing_anom['extreme_negative_splits'] - pacing_anom['extreme_positive_splits'],
            pacing_anom['extreme_positive_splits']
        ]
        colors = ['green', 'blue', 'red']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.7)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                        f'{count}\n({count/total*100:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Number of Runners')
        ax3.set_title('Pacing Anomalies', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Pacing anomaly data not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Residual analysis from trend decomposition
    if 'residual_analysis' in results['time_series_decomposition']:
        residual_analysis = results['time_series_decomposition']['residual_analysis']
        
        # Create synthetic residuals for visualization (in real implementation, would use actual residuals)
        sorted_df = df.sort_values('finish_time').reset_index(drop=True)
        positions = range(len(sorted_df))
        times = sorted_df['finish_time'].values
        
        # Fit trend and calculate residuals
        coeffs = np.polyfit(positions, times, 2)  # Quadratic trend
        trend = np.polyval(coeffs, positions)
        residuals = times - trend
        
        ax4.scatter(positions, residuals/60, alpha=0.6, s=20)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.axhline(y=np.std(residuals)/60, color='orange', linestyle='--', alpha=0.7, label='+1σ')
        ax4.axhline(y=-np.std(residuals)/60, color='orange', linestyle='--', alpha=0.7, label='-1σ')
        
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Residuals (minutes)')
        ax4.set_title('Trend Residuals Analysis', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_series_dashboard(df, results, output_dir):
    """Create a comprehensive time series analysis dashboard."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main performance progression
    ax1 = fig.add_subplot(gs[0, :2])
    
    sorted_df = df.sort_values('finish_time').reset_index(drop=True)
    sorted_df['position'] = range(1, len(sorted_df) + 1)
    
    ax1.scatter(sorted_df['position'], sorted_df['finish_time']/60, alpha=0.5, s=15)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Finish Time (minutes)')
    ax1.set_title('Performance Progression Through Race Field', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Pacing strategy pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    
    if 'pacing_analysis' in results and 'pacing_distribution' in results['pacing_analysis']:
        pacing_dist = results['pacing_analysis']['pacing_distribution']
        labels = list(pacing_dist.keys())
        sizes = list(pacing_dist.values())
        colors = ['green', 'blue', 'orange', 'red'][:len(labels)]
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Pacing Strategies', fontweight='bold')
    
    # Performance quartiles
    ax3 = fig.add_subplot(gs[0, 3])
    
    if 'performance_quartiles' in results['performance_trends']:
        quartiles = results['performance_trends']['performance_quartiles']
        q_names = ['Q25', 'Q50', 'Q75', 'Q90', 'Q95']
        q_times = [quartiles[f'q{q}']/60 for q in [25, 50, 75, 90, 95]]
        
        bars = ax3.bar(q_names, q_times, alpha=0.7, color='steelblue')
        ax3.set_ylabel('Time (minutes)')
        ax3.set_title('Performance Quartiles', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Age-performance relationship
    ax4 = fig.add_subplot(gs[1, :2])
    
    if 'age_performance_trend' in results['performance_trends']:
        age_trend = results['performance_trends']['age_performance_trend']
        ages = age_trend['ages']
        mean_times = [t/60 for t in age_trend['mean_times']]
        
        ax4.plot(ages, mean_times, 'bo-', alpha=0.7, markersize=6)
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Mean Finish Time (minutes)')
        ax4.set_title('Age-Performance Trend', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # Gap analysis
    ax5 = fig.add_subplot(gs[1, 2:])
    
    time_gaps = np.diff(sorted_df['finish_time'].values)
    ax5.plot(range(len(time_gaps)), time_gaps, alpha=0.7, linewidth=1)
    
    if 'gap_analysis' in results['performance_trends']:
        gap_analysis = results['performance_trends']['gap_analysis']
        ax5.axhline(y=gap_analysis['mean_gap'], color='red', linestyle='--', 
                   label=f'Mean Gap ({gap_analysis["mean_gap"]:.1f}s)')
    
    ax5.set_xlabel('Position')
    ax5.set_ylabel('Time Gap (seconds)')
    ax5.set_title('Time Gaps Between Consecutive Finishers', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
    TIME SERIES ANALYSIS SUMMARY
    
    Dataset Overview:
    • Total Participants: {len(df):,}
    • Performance Range: {df['finish_time'].min()/60:.1f} - {df['finish_time'].max()/60:.1f} minutes
    • Mean Finish Time: {df['finish_time'].mean()/60:.1f} minutes
    """
    
    if 'pacing_analysis' in results and 'negative_split_rate' in results['pacing_analysis']:
        neg_split_rate = results['pacing_analysis']['negative_split_rate']
        summary_text += f"• Negative Split Rate: {neg_split_rate:.1%}\n"
    
    if 'performance_trends' in results:
        perf_trends = results['performance_trends']['performance_distribution']
        summary_text += f"• Performance Skewness: {perf_trends['skewness']:.2f}\n"
        summary_text += f"• Performance Kurtosis: {perf_trends['kurtosis']:.2f}\n"
    
    if 'gap_analysis' in results['performance_trends']:
        gap_analysis = results['performance_trends']['gap_analysis']
        summary_text += f"• Mean Gap Between Runners: {gap_analysis['mean_gap']:.1f} seconds\n"
        summary_text += f"• Significant Performance Gaps: {gap_analysis['significant_gaps']['count']}\n"
    
    if 'outlier_analysis' in results and 'outlier_analysis' in results['outlier_analysis']:
        outlier_pct = results['outlier_analysis']['outlier_analysis']['iqr_outliers']['percentage']
        summary_text += f"• Performance Outliers: {outlier_pct:.1f}%\n"
    
    summary_text += f"\nKey Findings:\n"
    
    if 'pacing_analysis' in results and 'performance_by_pacing' in results['pacing_analysis']:
        best_strategy = min(results['pacing_analysis']['performance_by_pacing'].items(), 
                          key=lambda x: x[1]['mean_time'])[0]
        summary_text += f"• Best Pacing Strategy: {best_strategy}\n"
    
    if 'age_performance_trend' in results['performance_trends'] and 'optimal_age' in results['performance_trends']['age_performance_trend']:
        optimal_age = results['performance_trends']['age_performance_trend']['optimal_age']
        summary_text += f"• Optimal Performance Age: {optimal_age:.1f} years\n"
    
    summary_text += f"• Performance shows {'strong' if perf_trends.get('skewness', 0) > 1 else 'moderate'} positive skew\n"
    summary_text += f"• Time gaps suggest {'clustered' if gap_analysis.get('significant_gaps', {}).get('count', 0) > 10 else 'evenly distributed'} performance groups"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Time Series Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'time_series_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_time_series_report(df, results, output_dir):
    """Generate a comprehensive time series analysis report."""
    
    report = []
    report.append("TIME SERIES ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants\n")
    
    # Data summary
    report.append("DATA SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Participants: {len(df):,}")
    report.append(f"Performance Range: {df['finish_time'].min()/60:.1f} - {df['finish_time'].max()/60:.1f} minutes")
    report.append(f"Mean Finish Time: {df['finish_time'].mean()/60:.1f} minutes")
    report.append(f"Standard Deviation: {df['finish_time'].std()/60:.1f} minutes")
    
    if 'pacing_analysis' in results:
        with_splits = results['data_summary']['with_split_times']
        report.append(f"Participants with 10km splits: {with_splits}")
        if with_splits > 0:
            neg_split_rate = results['data_summary']['negative_split_rate']
            report.append(f"Negative split rate: {neg_split_rate:.1%}")
    
    report.append("")
    
    # Pacing analysis
    if 'pacing_analysis' in results and 'error' not in results['pacing_analysis']:
        report.append("PACING STRATEGY ANALYSIS")
        report.append("-" * 30)
        
        pacing_dist = results['pacing_analysis']['pacing_distribution']
        total_with_splits = sum(pacing_dist.values())
        
        for strategy, count in pacing_dist.items():
            percentage = count / total_with_splits * 100
            report.append(f"{strategy}: {count} runners ({percentage:.1f}%)")
        
        # Performance by strategy
        if 'performance_by_pacing' in results['pacing_analysis']:
            report.append("\nPerformance by Pacing Strategy:")
            perf_by_pacing = results['pacing_analysis']['performance_by_pacing']
            
            for strategy, stats in perf_by_pacing.items():
                mean_time = stats['mean_time'] / 60
                report.append(f"  {strategy}: {mean_time:.1f} minutes (n={stats['count']})")
        
        report.append("")
    
    # Performance trends
    if 'performance_trends' in results:
        report.append("PERFORMANCE TRENDS")
        report.append("-" * 20)
        
        perf_stats = results['performance_trends']['performance_distribution']
        report.append(f"Distribution characteristics:")
        report.append(f"  Skewness: {perf_stats['skewness']:.3f}")
        report.append(f"  Kurtosis: {perf_stats['kurtosis']:.3f}")
        
        # Quartile analysis
        if 'performance_quartiles' in results['performance_trends']:
            quartiles = results['performance_trends']['performance_quartiles']
            report.append(f"\nPerformance Quartiles:")
            for q, time in quartiles.items():
                report.append(f"  {q}: {time/60:.1f} minutes")
        
        # Age trends
        if 'age_performance_trend' in results['performance_trends']:
            age_trend = results['performance_trends']['age_performance_trend']
            if 'optimal_age' in age_trend:
                report.append(f"\nOptimal Performance Age: {age_trend['optimal_age']:.1f} years")
        
        report.append("")
    
    # Gap analysis
    if 'gap_analysis' in results['performance_trends']:
        report.append("TIME GAP ANALYSIS")
        report.append("-" * 20)
        
        gap_analysis = results['performance_trends']['gap_analysis']
        report.append(f"Mean gap between runners: {gap_analysis['mean_gap']:.1f} seconds")
        report.append(f"Median gap: {gap_analysis['median_gap']:.1f} seconds")
        report.append(f"Maximum gap: {gap_analysis['max_gap']:.1f} seconds")
        
        sig_gaps = gap_analysis['significant_gaps']
        report.append(f"Significant gaps (>{sig_gaps['threshold']:.1f}s): {sig_gaps['count']}")
        
        report.append("")
    
    # Outlier analysis
    if 'outlier_analysis' in results:
        report.append("OUTLIER ANALYSIS")
        report.append("-" * 18)
        
        if 'outlier_analysis' in results['outlier_analysis']:
            outlier_stats = results['outlier_analysis']['outlier_analysis']
            
            report.append("Statistical Outliers:")
            report.append(f"  Z-score method: {outlier_stats['z_score_outliers']['count']} ({outlier_stats['z_score_outliers']['percentage']:.1f}%)")
            report.append(f"  IQR method: {outlier_stats['iqr_outliers']['count']} ({outlier_stats['iqr_outliers']['percentage']:.1f}%)")
        
        if 'pacing_anomalies' in results['outlier_analysis']:
            pacing_anom = results['outlier_analysis']['pacing_anomalies']
            report.append(f"\nPacing Anomalies:")
            report.append(f"  Extreme negative splits: {pacing_anom['extreme_negative_splits']}")
            report.append(f"  Extreme positive splits: {pacing_anom['extreme_positive_splits']}")
            report.append(f"  Most extreme pace ratio: {pacing_anom['most_extreme_positive']:.2f}")
        
        report.append("")
    
    # Key findings and recommendations
    report.append("KEY FINDINGS & RECOMMENDATIONS")
    report.append("-" * 35)
    
    # Performance distribution insights
    if 'performance_trends' in results:
        skewness = results['performance_trends']['performance_distribution']['skewness']
        if skewness > 1:
            report.append("• Performance distribution shows strong positive skew - many slower finishers")
        elif skewness > 0.5:
            report.append("• Performance distribution shows moderate positive skew")
        else:
            report.append("• Performance distribution is relatively symmetric")
    
    # Pacing insights
    if 'pacing_analysis' in results and 'error' not in results['pacing_analysis']:
        neg_split_rate = results['pacing_analysis']['negative_split_rate']
        if neg_split_rate > 0.3:
            report.append("• High negative split rate suggests good pacing education")
        elif neg_split_rate < 0.1:
            report.append("• Low negative split rate suggests opportunity for pacing improvement")
        
        # Best pacing strategy
        if 'performance_by_pacing' in results['pacing_analysis']:
            best_strategy = min(results['pacing_analysis']['performance_by_pacing'].items(), 
                              key=lambda x: x[1]['mean_time'])[0]
            report.append(f"• Best performing pacing strategy: {best_strategy}")
    
    # Age trends
    if 'age_performance_trend' in results['performance_trends']:
        if 'optimal_age' in results['performance_trends']['age_performance_trend']:
            optimal_age = results['performance_trends']['age_performance_trend']['optimal_age']
            report.append(f"• Peak performance age: {optimal_age:.0f} years")
    
    # Gap analysis insights
    if 'gap_analysis' in results['performance_trends']:
        sig_gaps = results['performance_trends']['gap_analysis']['significant_gaps']['count']
        if sig_gaps > 10:
            report.append("• Multiple large performance gaps suggest distinct ability groups")
        else:
            report.append("• Relatively even performance distribution across the field")
    
    # Recommendations
    report.append("\nRecommendations:")
    report.append("• Use pacing analysis to develop runner education programs")
    report.append("• Consider age-specific performance targets and categories")
    report.append("• Monitor performance gaps to identify competitive balance")
    
    if 'pacing_analysis' in results and 'error' not in results['pacing_analysis']:
        neg_split_rate = results['pacing_analysis']['negative_split_rate']
        if neg_split_rate < 0.2:
            report.append("• Implement pacing strategy workshops to improve race execution")
    
    # Save report
    with open(output_dir / 'time_series_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating time series analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('time_series')
    
    # Load results
    df, results = load_results()
    
    # Create visualizations
    print("Plotting pacing analysis...")
    plot_pacing_analysis(df, results, output_dir)
    
    print("Plotting performance trends...")
    plot_performance_trends(df, results, output_dir)
    
    print("Plotting outlier analysis...")
    plot_outlier_analysis(df, results, output_dir)
    
    print("Creating time series dashboard...")
    plot_time_series_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_time_series_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()