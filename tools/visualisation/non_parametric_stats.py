"""
Non-parametric Statistics Visualizations
Comprehensive visualization suite for non-parametric statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, wilcoxon, spearmanr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class NonparametricVisualizer:
    """
    Comprehensive visualization suite for non-parametric statistical analysis.
    Creates publication-quality plots for robust statistical testing.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize non-parametric visualizer.
        
        Parameters:
        -----------
        style : str, default 'seaborn-v0_8'
            Matplotlib style
        figsize : Tuple[int, int], default (12, 8)
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_group_comparison_comprehensive(self, df: pd.DataFrame, 
                                          group_col: str, value_col: str,
                                          test_results: Dict = None,
                                          save_path: str = None) -> plt.Figure:
        """
        Comprehensive group comparison visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for analysis
        group_col : str
            Column defining groups
        value_col : str
            Column with values to compare
        test_results : Dict, optional
            Statistical test results to display
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Box plot with statistical annotations
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Create box plot
        box_plot = df.boxplot(column=value_col, by=group_col, ax=ax1, 
                             patch_artist=True, return_type='dict')
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(df[group_col].unique())))
        for patch, color in zip(box_plot[value_col]['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Box Plot Comparison')
        ax1.set_xlabel(group_col)
        ax1.set_ylabel(value_col)
        
        # Add statistical test results if provided
        if test_results:
            test_text = f"p-value: {test_results.get('p_value', 'N/A'):.4f}\n"
            test_text += f"Significant: {test_results.get('significant', 'N/A')}"
            ax1.text(0.02, 0.98, test_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
        
        # 2. Violin plot with individual points
        ax2 = fig.add_subplot(gs[0, 1])
        sns.violinplot(data=df, x=group_col, y=value_col, ax=ax2, alpha=0.7)
        sns.stripplot(data=df, x=group_col, y=value_col, ax=ax2, 
                     size=3, alpha=0.6, color='black')
        ax2.set_title('Violin Plot with Data Points')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Density plots by group
        ax3 = fig.add_subplot(gs[0, 2])
        groups = df[group_col].unique()
        for i, group in enumerate(groups):
            if pd.notna(group):
                group_data = df[df[group_col] == group][value_col].dropna()
                if len(group_data) > 1:
                    ax3.hist(group_data, alpha=0.6, density=True, 
                            label=f'{group} (n={len(group_data)})', 
                            color=colors[i], bins=20)
        
        ax3.set_xlabel(value_col)
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Density by Group')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q plots for normality assessment
        ax4 = fig.add_subplot(gs[1, 0])
        groups_list = [group for group in groups if pd.notna(group)]
        n_groups = len(groups_list)
        
        if n_groups > 0:
            group_data = df[df[group_col] == groups_list[0]][value_col].dropna()
            if len(group_data) > 3:
                stats.probplot(group_data, dist="norm", plot=ax4)
                ax4.set_title(f'Q-Q Plot: {groups_list[0]}')
                ax4.grid(True, alpha=0.3)
        
        # 5. Rank plots
        ax5 = fig.add_subplot(gs[1, 1])
        df_copy = df.copy()
        df_copy['rank'] = df_copy[value_col].rank()
        
        sns.boxplot(data=df_copy, x=group_col, y='rank', ax=ax5)
        ax5.set_title('Rank Comparison')
        ax5.set_ylabel('Rank')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Effect size visualization
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Calculate effect sizes between groups
        if len(groups_list) == 2:
            group1_data = df[df[group_col] == groups_list[0]][value_col].dropna()
            group2_data = df[df[group_col] == groups_list[1]][value_col].dropna()
            
            if len(group1_data) > 0 and len(group2_data) > 0:
                # Calculate Cohen's d equivalent for non-parametric
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                    (len(group2_data) - 1) * group2_data.var()) / 
                                   (len(group1_data) + len(group2_data) - 2))
                
                if pooled_std > 0:
                    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                    
                    # Visualization of effect size
                    x = np.linspace(min(group1_data.min(), group2_data.min()), 
                                  max(group1_data.max(), group2_data.max()), 100)
                    
                    # Plot distributions
                    ax6.hist(group1_data, alpha=0.5, density=True, label=groups_list[0], 
                            color=colors[0])
                    ax6.hist(group2_data, alpha=0.5, density=True, label=groups_list[1], 
                            color=colors[1])
                    
                    ax6.axvline(group1_data.mean(), color=colors[0], linestyle='--', 
                               label=f'{groups_list[0]} mean')
                    ax6.axvline(group2_data.mean(), color=colors[1], linestyle='--', 
                               label=f'{groups_list[1]} mean')
                    
                    ax6.set_title(f'Effect Size Visualization\nCohen\'s d = {cohens_d:.3f}')
                    ax6.legend()
        
        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        # Create summary statistics
        summary_stats = []
        for group in groups_list:
            group_data = df[df[group_col] == group][value_col].dropna()
            if len(group_data) > 0:
                stats_row = [
                    group,
                    len(group_data),
                    f"{group_data.mean():.2f}",
                    f"{group_data.median():.2f}",
                    f"{group_data.std():.2f}",
                    f"{group_data.quantile(0.25):.2f}",
                    f"{group_data.quantile(0.75):.2f}",
                    f"{group_data.min():.2f}",
                    f"{group_data.max():.2f}"
                ]
                summary_stats.append(stats_row)
        
        table_data = pd.DataFrame(summary_stats, 
                                columns=['Group', 'N', 'Mean', 'Median', 'Std', 
                                       'Q1', 'Q3', 'Min', 'Max'])
        
        table = ax7.table(cellText=table_data.values,
                         colLabels=table_data.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#E6E6FA')
        
        ax7.set_title('Summary Statistics by Group', y=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_analysis(self, df: pd.DataFrame, 
                                 x_col: str, y_col: str,
                                 correlation_results: Dict = None,
                                 save_path: str = None) -> plt.Figure:
        """
        Comprehensive correlation analysis visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for analysis
        x_col : str
            X variable column
        y_col : str
            Y variable column
        correlation_results : Dict, optional
            Correlation test results
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Remove missing values
        clean_data = df[[x_col, y_col]].dropna()
        x_data = clean_data[x_col]
        y_data = clean_data[y_col]
        
        # 1. Scatter plot with regression lines
        ax1.scatter(x_data, y_data, alpha=0.6, s=50)
        
        # Add regression lines
        z_parametric = np.polyfit(x_data, y_data, 1)
        p_parametric = np.poly1d(z_parametric)
        ax1.plot(x_data, p_parametric(x_data), "r--", alpha=0.8, label='Linear fit')
        
        # Non-parametric (lowess) fit
        from scipy.interpolate import UnivariateSpline
        if len(x_data) > 3:
            sorted_idx = np.argsort(x_data)
            x_sorted = x_data.iloc[sorted_idx]
            y_sorted = y_data.iloc[sorted_idx]
            
            try:
                spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_data))
                ax1.plot(x_sorted, spline(x_sorted), "g-", alpha=0.8, label='Spline fit')
            except:
                pass
        
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title('Scatter Plot with Trend Lines')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add correlation information
        if correlation_results:
            corr_text = f"Spearman ρ: {correlation_results.get('spearman_rho', 'N/A'):.3f}\n"
            corr_text += f"p-value: {correlation_results.get('spearman_p_value', 'N/A'):.4f}\n"
            corr_text += f"Kendall τ: {correlation_results.get('kendall_tau', 'N/A'):.3f}"
            ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
        
        # 2. Rank correlation plot
        x_ranks = x_data.rank()
        y_ranks = y_data.rank()
        
        ax2.scatter(x_ranks, y_ranks, alpha=0.6, s=50, color='orange')
        
        # Add rank regression line
        z_rank = np.polyfit(x_ranks, y_ranks, 1)
        p_rank = np.poly1d(z_rank)
        ax2.plot(x_ranks, p_rank(x_ranks), "r--", alpha=0.8)
        
        ax2.set_xlabel(f'{x_col} Ranks')
        ax2.set_ylabel(f'{y_col} Ranks')
        ax2.set_title('Rank Correlation Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals from parametric fit
        residuals = y_data - p_parametric(x_data)
        
        ax3.scatter(p_parametric(x_data), residuals, alpha=0.6, s=50, color='red')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals vs Fitted')
        ax3.grid(True, alpha=0.3)
        
        # 4. Marginal distributions
        ax4.hist(x_data, alpha=0.5, bins=20, orientation='horizontal', 
                color='blue', label=x_col, density=True)
        ax4_twin = ax4.twinx()
        ax4_twin.hist(y_data, alpha=0.5, bins=20, 
                     color='orange', label=y_col, density=True)
        
        ax4.set_xlabel('Density')
        ax4.set_ylabel(x_col)
        ax4_twin.set_ylabel(y_col)
        ax4.set_title('Marginal Distributions')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_percentile_analysis(self, df: pd.DataFrame, 
                               value_col: str, group_col: str = None,
                               percentiles: List[float] = None,
                               save_path: str = None) -> plt.Figure:
        """
        Comprehensive percentile analysis visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for analysis
        value_col : str
            Column to analyze
        group_col : str, optional
            Column to group by
        percentiles : List[float], optional
            Percentiles to display
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        if percentiles is None:
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        if group_col is None:
            # Overall analysis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            data = df[value_col].dropna()
            
            # 1. Percentile plot
            percentile_values = [data.quantile(p/100) for p in percentiles]
            
            ax1.plot(percentiles, percentile_values, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Percentile')
            ax1.set_ylabel(value_col)
            ax1.set_title('Percentile Plot')
            ax1.grid(True, alpha=0.3)
            
            # Add percentile labels
            for i, (p, v) in enumerate(zip(percentiles, percentile_values)):
                ax1.annotate(f'{v:.1f}', (p, v), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # 2. Box plot with percentiles
            box_plot = ax2.boxplot(data, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][0].set_alpha(0.7)
            
            # Add percentile lines
            for p in [5, 95]:
                percentile_val = data.quantile(p/100)
                ax2.axhline(y=percentile_val, color='red', linestyle='--', 
                           alpha=0.7, label=f'{p}th percentile')
            
            ax2.set_ylabel(value_col)
            ax2.set_title('Box Plot with Extreme Percentiles')
            ax2.legend()
            
            # 3. Cumulative distribution
            sorted_data = np.sort(data)
            cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            ax3.plot(sorted_data, cumulative_prob * 100, linewidth=2)
            ax3.set_xlabel(value_col)
            ax3.set_ylabel('Cumulative Percentage')
            ax3.set_title('Cumulative Distribution Function')
            ax3.grid(True, alpha=0.3)
            
            # Add percentile reference lines
            for p in [25, 50, 75]:
                percentile_val = data.quantile(p/100)
                ax3.axvline(x=percentile_val, color='red', linestyle='--', alpha=0.5)
                ax3.axhline(y=p, color='red', linestyle='--', alpha=0.5)
            
            # 4. Percentile differences
            percentile_diffs = np.diff(percentile_values)
            ax4.bar(range(len(percentile_diffs)), percentile_diffs, alpha=0.7)
            ax4.set_xlabel('Percentile Interval')
            ax4.set_ylabel('Value Difference')
            ax4.set_title('Differences Between Consecutive Percentiles')
            ax4.set_xticks(range(len(percentile_diffs)))
            ax4.set_xticklabels([f'{percentiles[i]}-{percentiles[i+1]}' 
                               for i in range(len(percentile_diffs))], rotation=45)
            ax4.grid(True, alpha=0.3)
            
        else:
            # Group-wise analysis
            groups = df[group_col].unique()
            n_groups = len([g for g in groups if pd.notna(g)])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.ravel()
            
            # 1. Percentile comparison across groups
            ax1 = axes[0]
            for i, group in enumerate(groups):
                if pd.notna(group):
                    group_data = df[df[group_col] == group][value_col].dropna()
                    if len(group_data) > 0:
                        group_percentiles = [group_data.quantile(p/100) for p in percentiles]
                        ax1.plot(percentiles, group_percentiles, 'o-', 
                               linewidth=2, markersize=6, label=f'{group} (n={len(group_data)})')
            
            ax1.set_xlabel('Percentile')
            ax1.set_ylabel(value_col)
            ax1.set_title('Percentile Comparison by Group')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Box plots by group
            ax2 = axes[1]
            df.boxplot(column=value_col, by=group_col, ax=ax2)
            ax2.set_title('Box Plot Comparison by Group')
            
            # 3. Percentile spread analysis
            ax3 = axes[2]
            iqr_values = []
            group_names = []
            
            for group in groups:
                if pd.notna(group):
                    group_data = df[df[group_col] == group][value_col].dropna()
                    if len(group_data) > 0:
                        q75 = group_data.quantile(0.75)
                        q25 = group_data.quantile(0.25)
                        iqr_values.append(q75 - q25)
                        group_names.append(group)
            
            if iqr_values:
                bars = ax3.bar(group_names, iqr_values, alpha=0.7)
                ax3.set_ylabel('Interquartile Range')
                ax3.set_xlabel(group_col)
                ax3.set_title('IQR Comparison by Group')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, iqr_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.1f}', ha='center', va='bottom')
            
            # 4. Extreme percentiles comparison
            ax4 = axes[3]
            extreme_percentiles = [5, 95]
            
            x_pos = np.arange(len(group_names))
            width = 0.35
            
            for i, percentile in enumerate(extreme_percentiles):
                percentile_values = []
                for group in group_names:
                    group_data = df[df[group_col] == group][value_col].dropna()
                    if len(group_data) > 0:
                        percentile_values.append(group_data.quantile(percentile/100))
                    else:
                        percentile_values.append(0)
                
                ax4.bar(x_pos + i * width, percentile_values, width, 
                       alpha=0.7, label=f'{percentile}th percentile')
            
            ax4.set_xlabel(group_col)
            ax4.set_ylabel(value_col)
            ax4.set_title('Extreme Percentiles by Group')
            ax4.set_xticks(x_pos + width / 2)
            ax4.set_xticklabels(group_names, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_rank_analysis(self, df: pd.DataFrame, value_col: str,
                          group_col: str = None, save_path: str = None) -> plt.Figure:
        """
        Comprehensive rank analysis visualization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for analysis
        value_col : str
            Column to rank
        group_col : str, optional
            Column to group by
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate ranks
        df_copy = df.copy()
        df_copy['rank'] = df_copy[value_col].rank(method='min')
        df_copy['percentile_rank'] = df_copy[value_col].rank(pct=True)
        
        # 1. Value vs Rank scatter
        ax1.scatter(df_copy[value_col], df_copy['rank'], alpha=0.6, s=50)
        ax1.set_xlabel(value_col)
        ax1.set_ylabel('Rank')
        ax1.set_title('Value vs Rank Relationship')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df_copy[value_col].dropna(), 
                      df_copy['rank'].dropna(), 1)
        p = np.poly1d(z)
        ax1.plot(df_copy[value_col], p(df_copy[value_col]), "r--", alpha=0.8)
        
        # 2. Rank distribution
        ax2.hist(df_copy['rank'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Rank Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Percentile rank analysis
        if group_col and group_col in df_copy.columns:
            # Group-wise percentile ranks
            groups = df_copy[group_col].unique()
            for group in groups:
                if pd.notna(group):
                    group_data = df_copy[df_copy[group_col] == group]
                    ax3.hist(group_data['percentile_rank'], alpha=0.6, 
                           label=f'{group} (n={len(group_data)})', bins=20)
            
            ax3.set_xlabel('Percentile Rank')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Percentile Rank Distribution by Group')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Mean ranks by group
            mean_ranks = df_copy.groupby(group_col)['rank'].mean().sort_values()
            bars = ax4.bar(range(len(mean_ranks)), mean_ranks.values, alpha=0.7)
            ax4.set_xlabel(group_col)
            ax4.set_ylabel('Mean Rank')
            ax4.set_title('Mean Rank by Group')
            ax4.set_xticks(range(len(mean_ranks)))
            ax4.set_xticklabels(mean_ranks.index, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, mean_ranks.values)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}', ha='center', va='bottom')
        else:
            # Overall percentile rank
            ax3.hist(df_copy['percentile_rank'], bins=30, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Percentile Rank')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Percentile Rank Distribution')
            ax3.grid(True, alpha=0.3)
            
            # 4. Rank vs Percentile
            ax4.scatter(df_copy['rank'], df_copy['percentile_rank'], alpha=0.6, s=50)
            ax4.set_xlabel('Rank')
            ax4.set_ylabel('Percentile Rank')
            ax4.set_title('Rank vs Percentile Rank')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_nonparametric_dashboard(self, df: pd.DataFrame, 
                                     value_col: str, group_col: str,
                                     test_results: Dict = None,
                                     save_dir: str = 'nonparametric_visualizations/') -> Dict[str, plt.Figure]:
        """
        Create a complete dashboard of non-parametric visualizations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for analysis
        value_col : str
            Column with values to analyze
        group_col : str
            Column defining groups
        test_results : Dict, optional
            Statistical test results
        save_dir : str, default 'nonparametric_visualizations/'
            Directory to save plots
            
        Returns:
        --------
        Dict[str, plt.Figure]
            Dictionary of generated figures
        """
        import os
        
        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        print("Creating non-parametric analysis dashboard...")
        
        # 1. Group comparison analysis
        print("  - Group comparison analysis...")
        fig1 = self.plot_group_comparison_comprehensive(
            df, group_col, value_col, test_results,
            save_path=os.path.join(save_dir, 'group_comparison.png') if save_dir else None
        )
        figures['group_comparison'] = fig1
        
        # 2. Percentile analysis
        print("  - Percentile analysis...")
        fig2 = self.plot_percentile_analysis(
            df, value_col, group_col,
            save_path=os.path.join(save_dir, 'percentile_analysis.png') if save_dir else None
        )
        figures['percentile_analysis'] = fig2
        
        # 3. Rank analysis
        print("  - Rank analysis...")
        fig3 = self.plot_rank_analysis(
            df, value_col, group_col,
            save_path=os.path.join(save_dir, 'rank_analysis.png') if save_dir else None
        )
        figures['rank_analysis'] = fig3
        
        # 4. Correlation analysis (if we have a second numeric column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2 and value_col in numeric_cols:
            other_col = [col for col in numeric_cols if col != value_col][0]
            
            print(f"  - Correlation analysis ({value_col} vs {other_col})...")
            fig4 = self.plot_correlation_analysis(
                df, other_col, value_col,
                save_path=os.path.join(save_dir, 'correlation_analysis.png') if save_dir else None
            )
            figures['correlation_analysis'] = fig4
        
        print(f"✓ Non-parametric dashboard complete! {len(figures)} visualizations created.")
        
        return figures


def main():
    """Example usage of non-parametric visualizations."""
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    # Generate realistic race data with different distributions
    elite_times = np.random.gamma(2, 20, 100) + 120  # Faster, more consistent
    regular_times = np.random.normal(200, 30, 250)  # Normal distribution
    recreational_times = np.random.exponential(40, 150) + 180  # Right-skewed
    
    # Combine data
    all_times = np.concatenate([elite_times, regular_times, recreational_times])
    groups = ['Elite'] * 100 + ['Regular'] * 250 + ['Recreational'] * 150
    ages = np.random.randint(20, 70, n_samples)
    
    df = pd.DataFrame({
        'Time': all_times,
        'Group': groups,
        'Age': ages,
        'Experience': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_samples)
    })
    
    # Initialize visualizer
    visualizer = NonparametricVisualizer()
    
    # Create dashboard
    figures = visualizer.create_nonparametric_dashboard(
        df, 'Time', 'Group', save_dir='nonparametric_demo_plots/'
    )
    
    print("Non-parametric visualization demo completed!")
    print(f"Generated {len(figures)} visualizations in 'nonparametric_demo_plots/' directory")


if __name__ == "__main__":
    main()