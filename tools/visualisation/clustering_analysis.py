#!/usr/bin/env python3
"""
Clustering Analysis Visualization
Creates comprehensive visualizations for K-means and DBSCAN clustering results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
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
    """Load clustering analysis results and data."""
    # Load processed data with cluster assignments
    df = pd.read_csv('data/processed/clustering_data.csv')
    
    # Load results
    with open('data/processed/clustering_results.json', 'r') as f:
        results = json.load(f)
    
    # Load model objects
    try:
        with open('data/processed/clustering_models.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}
    
    return df, results, models

def plot_cluster_visualization(df, results, output_dir):
    """Plot cluster visualizations in 2D and 3D."""
    fig = plt.figure(figsize=(20, 12))
    
    # K-means clustering visualization
    # Plot 1: 2D scatter - Age vs Performance (K-means)
    ax1 = plt.subplot(2, 4, 1)
    
    unique_clusters = df['kmeans_cluster'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_data = df[df['kmeans_cluster'] == cluster]
        ax1.scatter(cluster_data['age'], cluster_data['finish_time']/60, 
                   c=[color], label=f'Cluster {cluster}', alpha=0.6, s=30)
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Finish Time (minutes)')
    ax1.set_title('K-means Clusters: Age vs Performance', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2D scatter - Performance Percentile vs Age-graded (K-means)
    ax2 = plt.subplot(2, 4, 2)
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_data = df[df['kmeans_cluster'] == cluster]
        ax2.scatter(cluster_data['performance_percentile'], cluster_data['age_graded_time']/60, 
                   c=[color], label=f'Cluster {cluster}', alpha=0.6, s=30)
    
    ax2.set_xlabel('Performance Percentile')
    ax2.set_ylabel('Age-Graded Time (minutes)')
    ax2.set_title('K-means: Performance vs Age-Graded Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DBSCAN clustering
    ax3 = plt.subplot(2, 4, 3)
    
    unique_dbscan = df['dbscan_cluster'].unique()
    dbscan_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_dbscan)))
    
    for cluster, color in zip(unique_dbscan, dbscan_colors):
        cluster_data = df[df['dbscan_cluster'] == cluster]
        label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
        marker = 'x' if cluster == -1 else 'o'
        alpha = 0.3 if cluster == -1 else 0.6
        
        ax3.scatter(cluster_data['age'], cluster_data['finish_time']/60, 
                   c=[color], label=label, alpha=alpha, s=30, marker=marker)
    
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Finish Time (minutes)')
    ax3.set_title('DBSCAN Clusters: Age vs Performance', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PCA visualization (if available)
    ax4 = plt.subplot(2, 4, 4)
    
    if 'pca' in results and 'transformed_data' in results['pca']:
        pca_data = np.array(results['pca']['transformed_data'])
        
        for cluster, color in zip(unique_clusters, colors):
            cluster_mask = df['kmeans_cluster'] == cluster
            cluster_pca = pca_data[cluster_mask]
            ax4.scatter(cluster_pca[:, 0], cluster_pca[:, 1], 
                       c=[color], label=f'Cluster {cluster}', alpha=0.6, s=30)
        
        ax4.set_xlabel('First Principal Component')
        ax4.set_ylabel('Second Principal Component')
        ax4.set_title('K-means Clusters in PCA Space', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'PCA data not available', ha='center', va='center', 
                transform=ax4.transAxes)
    
    # Plot 5: Cluster size comparison
    ax5 = plt.subplot(2, 4, 5)
    
    kmeans_sizes = df['kmeans_cluster'].value_counts().sort_index()
    dbscan_sizes = df['dbscan_cluster'].value_counts().sort_index()
    
    x = np.arange(len(kmeans_sizes))
    width = 0.35
    
    ax5.bar(x - width/2, kmeans_sizes.values, width, label='K-means', alpha=0.7)
    
    # Adjust DBSCAN data for comparison
    dbscan_labels = [f'C{i}' if i != -1 else 'Noise' for i in dbscan_sizes.index]
    ax5_twin = ax5.twinx()
    ax5_twin.bar(x + width/2, dbscan_sizes.values, width, label='DBSCAN', 
                alpha=0.7, color='orange')
    
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('K-means Cluster Size', color='blue')
    ax5_twin.set_ylabel('DBSCAN Cluster Size', color='orange')
    ax5.set_title('Cluster Size Comparison', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'C{i}' for i in kmeans_sizes.index])
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Gender distribution by cluster
    ax6 = plt.subplot(2, 4, 6)
    
    gender_cluster = pd.crosstab(df['kmeans_cluster'], df['gender'], normalize='index') * 100
    gender_cluster.plot(kind='bar', ax=ax6, alpha=0.7)
    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Percentage')
    ax6.set_title('Gender Distribution by K-means Cluster', fontweight='bold')
    ax6.legend(title='Gender')
    ax6.tick_params(axis='x', rotation=0)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Club membership by cluster
    ax7 = plt.subplot(2, 4, 7)
    
    club_cluster = df.groupby('kmeans_cluster')['has_club'].mean() * 100
    ax7.bar(club_cluster.index, club_cluster.values, alpha=0.7, color='green')
    ax7.set_xlabel('Cluster')
    ax7.set_ylabel('Club Membership (%)')
    ax7.set_title('Club Membership by Cluster', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Performance consistency by cluster
    ax8 = plt.subplot(2, 4, 8)
    
    if 'pace_consistency' in df.columns:
        pace_cluster = df.groupby('kmeans_cluster')['pace_consistency'].mean()
        ax8.bar(pace_cluster.index, pace_cluster.values, alpha=0.7, color='purple')
        ax8.set_xlabel('Cluster')
        ax8.set_ylabel('Mean Pace Consistency')
        ax8.set_title('Pace Consistency by Cluster', fontweight='bold')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'Pace data not available', ha='center', va='center', 
                transform=ax8.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_analysis(df, results, output_dir):
    """Plot detailed cluster analysis and characteristics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: K-means cluster characteristics heatmap
    if 'kmeans_clusters' in results['analysis']:
        cluster_stats = []
        cluster_names = []
        
        for cluster_id, stats in results['analysis']['kmeans_clusters'].items():
            cluster_names.append(cluster_id.replace('cluster_', 'C'))
            cluster_stats.append([
                stats['mean_finish_time'] / 60,  # Convert to minutes
                stats['mean_age'],
                stats['club_membership_rate'] * 100,
                stats['mean_performance_percentile']
            ])
        
        cluster_df = pd.DataFrame(cluster_stats, 
                                 columns=['Finish Time (min)', 'Age', 'Club Rate (%)', 'Perf. Percentile'],
                                 index=cluster_names)
        
        # Normalize for heatmap
        cluster_df_norm = (cluster_df - cluster_df.min()) / (cluster_df.max() - cluster_df.min())
        
        sns.heatmap(cluster_df_norm.T, annot=cluster_df.T, fmt='.1f', 
                   cmap='RdYlBu_r', ax=ax1, cbar_kws={'label': 'Normalized Score'})
        ax1.set_title('K-means Cluster Characteristics', fontweight='bold')
        ax1.set_xlabel('Cluster')
    
    # Plot 2: Model quality metrics
    metrics_data = {
        'K-means': {
            'Silhouette Score': results['kmeans']['silhouette_score'],
            'Calinski-Harabasz': results['kmeans']['calinski_harabasz_score'] / 1000,  # Scale down
            'Davies-Bouldin': 2 - results['kmeans']['davies_bouldin_score']  # Invert (lower is better)
        }
    }
    
    if results['dbscan']['silhouette_score'] is not None:
        metrics_data['DBSCAN'] = {
            'Silhouette Score': results['dbscan']['silhouette_score'],
            'Noise Ratio': 1 - results['dbscan']['noise_ratio'],  # Invert (lower noise is better)
            'N Clusters': results['dbscan']['n_clusters'] / 10  # Normalize
        }
    
    metrics_df = pd.DataFrame(metrics_data).fillna(0)
    metrics_df.plot(kind='bar', ax=ax2, alpha=0.7)
    ax2.set_title('Clustering Quality Metrics', fontweight='bold')
    ax2.set_ylabel('Score (normalized)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimal K selection (K-means)
    if 'k_selection_metrics' in results['kmeans']:
        k_metrics = results['kmeans']['k_selection_metrics']
        k_range = k_metrics['k_range']
        
        ax3_twin = ax3.twinx()
        
        # Silhouette scores
        line1 = ax3.plot(k_range, k_metrics['silhouette_scores'], 'bo-', 
                        label='Silhouette Score', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Silhouette Score', color='blue')
        ax3.set_title('Optimal K Selection for K-means', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Inertia (elbow method)
        line2 = ax3_twin.plot(k_range, k_metrics['inertias'], 'ro-', 
                             label='Inertia', linewidth=2, markersize=6)
        ax3_twin.set_ylabel('Inertia', color='red')
        
        # Highlight optimal k
        optimal_k = results['kmeans']['optimal_k']
        ax3.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, 
                   label=f'Optimal k={optimal_k}')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines] + ['Optimal k']
        ax3.legend(lines + [plt.Line2D([0], [0], color='green', linestyle='--')], 
                  labels, loc='upper right')
    
    # Plot 4: Cluster interpretation
    ax4.axis('off')
    
    if 'kmeans_interpretations' in results['analysis']:
        interpretations = results['analysis']['kmeans_interpretations']
        
        interpretation_text = "CLUSTER INTERPRETATIONS\n" + "="*25 + "\n\n"
        
        for cluster_id, interp in interpretations.items():
            cluster_num = cluster_id.replace('cluster_', '')
            interpretation_text += f"Cluster {cluster_num}: {interp['label']}\n"
            interpretation_text += f"  Performance: {interp['performance_level']}\n"
            interpretation_text += f"  Age Group: {interp['age_group']}\n"
            if interp['key_characteristics']:
                interpretation_text += f"  Characteristics: {', '.join(interp['key_characteristics'])}\n"
            interpretation_text += "\n"
        
        # Add cluster sizes
        if 'kmeans_clusters' in results['analysis']:
            interpretation_text += "CLUSTER SIZES\n" + "-"*15 + "\n"
            for cluster_id, stats in results['analysis']['kmeans_clusters'].items():
                cluster_num = cluster_id.replace('cluster_', '')
                interpretation_text += f"Cluster {cluster_num}: {stats['size']} runners ({stats['percentage']:.1f}%)\n"
        
        ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_cluster_visualization(df, results, output_dir):
    """Create 3D visualization of clusters."""
    fig = plt.figure(figsize=(15, 6))
    
    # 3D plot for K-means
    ax1 = fig.add_subplot(121, projection='3d')
    
    unique_clusters = df['kmeans_cluster'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_data = df[df['kmeans_cluster'] == cluster]
        ax1.scatter(cluster_data['age'], 
                   cluster_data['finish_time']/60, 
                   cluster_data['performance_percentile'],
                   c=[color], label=f'Cluster {cluster}', alpha=0.6, s=30)
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Finish Time (min)')
    ax1.set_zlabel('Performance Percentile')
    ax1.set_title('3D K-means Clusters', fontweight='bold')
    ax1.legend()
    
    # 3D plot for DBSCAN
    ax2 = fig.add_subplot(122, projection='3d')
    
    unique_dbscan = df['dbscan_cluster'].unique()
    dbscan_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_dbscan)))
    
    for cluster, color in zip(unique_dbscan, dbscan_colors):
        cluster_data = df[df['dbscan_cluster'] == cluster]
        label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
        marker = 'x' if cluster == -1 else 'o'
        alpha = 0.3 if cluster == -1 else 0.6
        
        ax2.scatter(cluster_data['age'], 
                   cluster_data['finish_time']/60, 
                   cluster_data['performance_percentile'],
                   c=[color], label=label, alpha=alpha, s=30, marker=marker)
    
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Finish Time (min)')
    ax2.set_zlabel('Performance Percentile')
    ax2.set_title('3D DBSCAN Clusters', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_analysis(results, output_dir):
    """Plot PCA analysis results."""
    if 'pca' not in results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    pca_results = results['pca']
    
    # Plot 1: Explained variance ratio
    explained_var = pca_results['explained_variance_ratio']
    cumulative_var = pca_results['cumulative_variance']
    
    n_components = len(explained_var)
    x = range(1, min(n_components + 1, 21))  # Show first 20 components
    
    ax1.bar(x, explained_var[:20], alpha=0.7, color='skyblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA: Explained Variance by Component', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    ax2.plot(x, cumulative_var[:20], 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
    ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    
    # Mark important points
    n_95 = pca_results['n_components_95']
    if n_95 <= 20:
        ax2.axvline(x=n_95, color='red', linestyle=':', alpha=0.7)
        ax2.text(n_95 + 0.5, 0.85, f'{n_95} components\nfor 95%', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('PCA: Cumulative Explained Variance', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature loadings for first two components
    if 'feature_loadings' in pca_results and len(pca_results['feature_loadings']) >= 2:
        loadings = np.array(pca_results['feature_loadings'])
        
        # Get feature names (this would need to be stored with the results)
        # For now, create generic names
        n_features = loadings.shape[1]
        feature_names = [f'Feature_{i+1}' for i in range(min(n_features, 10))]
        
        pc1_loadings = loadings[0, :len(feature_names)]
        pc2_loadings = loadings[1, :len(feature_names)]
        
        ax3.scatter(pc1_loadings, pc2_loadings, alpha=0.7, s=60)
        
        # Add feature labels
        for i, name in enumerate(feature_names):
            ax3.annotate(name, (pc1_loadings[i], pc2_loadings[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('PC1 Loadings')
        ax3.set_ylabel('PC2 Loadings')
        ax3.set_title('PCA: Feature Loadings (PC1 vs PC2)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Feature loadings not available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: PCA summary statistics
    ax4.axis('off')
    
    pca_text = f"""
    PCA ANALYSIS SUMMARY
    
    Total Components: {len(explained_var)}
    Components for 95% variance: {pca_results['n_components_95']}
    
    First 5 Components:
    """
    
    for i in range(min(5, len(explained_var))):
        pca_text += f"  PC{i+1}: {explained_var[i]:.3f} ({explained_var[i]*100:.1f}%)\n"
    
    pca_text += f"\n  Cumulative (PC1-5): {sum(explained_var[:5]):.3f} ({sum(explained_var[:5])*100:.1f}%)"
    
    ax4.text(0.1, 0.9, pca_text, transform=ax4.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_clustering_dashboard(df, results, output_dir):
    """Create a comprehensive clustering analysis dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main cluster visualization
    ax1 = fig.add_subplot(gs[0, :2])
    
    unique_clusters = df['kmeans_cluster'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        cluster_data = df[df['kmeans_cluster'] == cluster]
        ax1.scatter(cluster_data['age'], cluster_data['finish_time']/60, 
                   c=[color], label=f'Cluster {cluster}', alpha=0.6, s=40)
    
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Finish Time (minutes)')
    ax1.set_title('K-means Clusters: Age vs Performance', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cluster characteristics
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if 'kmeans_clusters' in results['analysis']:
        cluster_stats = []
        cluster_names = []
        
        for cluster_id, stats in results['analysis']['kmeans_clusters'].items():
            cluster_names.append(cluster_id.replace('cluster_', 'C'))
            cluster_stats.append([
                stats['mean_finish_time'] / 60,
                stats['mean_age'],
                stats['club_membership_rate'] * 100,
                stats['mean_performance_percentile']
            ])
        
        cluster_df = pd.DataFrame(cluster_stats, 
                                 columns=['Time (min)', 'Age', 'Club %', 'Perf %'],
                                 index=cluster_names)
        
        cluster_df_norm = (cluster_df - cluster_df.min()) / (cluster_df.max() - cluster_df.min())
        
        sns.heatmap(cluster_df_norm.T, annot=cluster_df.T, fmt='.1f', 
                   cmap='RdYlBu_r', ax=ax2, cbar_kws={'label': 'Score'})
        ax2.set_title('Cluster Characteristics Heatmap', fontsize=14, fontweight='bold')
    
    # Cluster sizes
    ax3 = fig.add_subplot(gs[1, 0])
    
    cluster_sizes = df['kmeans_cluster'].value_counts().sort_index()
    cluster_percentages = (cluster_sizes / len(df) * 100).round(1)
    
    wedges, texts, autotexts = ax3.pie(cluster_sizes.values, 
                                      labels=[f'C{i}\n({cluster_percentages[i]}%)' for i in cluster_sizes.index],
                                      autopct='%1.0f', startangle=90)
    ax3.set_title('Cluster Size Distribution', fontweight='bold')
    
    # Model quality metrics
    ax4 = fig.add_subplot(gs[1, 1])
    
    metrics = ['Silhouette\nScore', 'Calinski\nHarabasz', 'Davies\nBouldin']
    kmeans_values = [
        results['kmeans']['silhouette_score'],
        results['kmeans']['calinski_harabasz_score'] / 1000,  # Normalize
        2 - results['kmeans']['davies_bouldin_score']  # Invert
    ]
    
    ax4.bar(metrics, kmeans_values, alpha=0.7, color='steelblue')
    ax4.set_ylabel('Score (normalized)')
    ax4.set_title('K-means Quality Metrics', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Gender distribution by cluster
    ax5 = fig.add_subplot(gs[1, 2])
    
    gender_dist = pd.crosstab(df['kmeans_cluster'], df['gender'], normalize='index') * 100
    gender_dist.plot(kind='bar', ax=ax5, alpha=0.7)
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Percentage')
    ax5.set_title('Gender Distribution', fontweight='bold')
    ax5.legend(title='Gender')
    ax5.tick_params(axis='x', rotation=0)
    
    # PCA explained variance
    ax6 = fig.add_subplot(gs[1, 3])
    
    if 'pca' in results:
        explained_var = results['pca']['explained_variance_ratio'][:10]
        ax6.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='orange')
        ax6.set_xlabel('Principal Component')
        ax6.set_ylabel('Explained Variance')
        ax6.set_title('PCA: Top 10 Components', fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    # Summary statistics and interpretations
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create summary text
    summary_text = f"""
    CLUSTERING ANALYSIS SUMMARY
    
    Dataset Overview:
    • Total Participants: {len(df):,}
    • K-means Clusters: {results['kmeans']['optimal_k']}
    • DBSCAN Clusters: {results['dbscan']['n_clusters']} (+ {results['dbscan']['n_noise']} noise points)
    
    K-means Performance:
    • Silhouette Score: {results['kmeans']['silhouette_score']:.3f}
    • Optimal K selected by: Highest silhouette score
    
    Cluster Interpretations:
    """
    
    if 'kmeans_interpretations' in results['analysis']:
        for cluster_id, interp in results['analysis']['kmeans_interpretations'].items():
            cluster_num = cluster_id.replace('cluster_', '')
            summary_text += f"• Cluster {cluster_num}: {interp['label']}\n"
    
    if 'pca' in results:
        summary_text += f"\nDimensionality Reduction:\n"
        summary_text += f"• {results['pca']['n_components_95']} components explain 95% of variance\n"
        summary_text += f"• First 3 components explain {sum(results['pca']['explained_variance_ratio'][:3])*100:.1f}% of variance"
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Clustering Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'clustering_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_clustering_report(df, results, output_dir):
    """Generate a comprehensive clustering analysis report."""
    report = []
    report.append("CLUSTERING ANALYSIS REPORT")
    report.append("=" * 50)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} participants\n")
    
    # Data summary
    report.append("DATA SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Participants: {len(df):,}")
    report.append(f"Features Used: {results['data_summary']['features_used']}")
    report.append(f"Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    report.append(f"Performance Range: {df['finish_time'].min()/60:.1f} - {df['finish_time'].max()/60:.1f} minutes")
    report.append("")
    
    # K-means results
    report.append("K-MEANS CLUSTERING RESULTS")
    report.append("-" * 30)
    report.append(f"Optimal Number of Clusters: {results['kmeans']['optimal_k']}")
    report.append(f"Silhouette Score: {results['kmeans']['silhouette_score']:.3f}")
    report.append(f"Calinski-Harabasz Score: {results['kmeans']['calinski_harabasz_score']:.2f}")
    report.append(f"Davies-Bouldin Score: {results['kmeans']['davies_bouldin_score']:.3f}")
    report.append("")
    
    # Cluster characteristics
    if 'kmeans_clusters' in results['analysis']:
        report.append("CLUSTER CHARACTERISTICS")
        report.append("-" * 25)
        
        for cluster_id, stats in results['analysis']['kmeans_clusters'].items():
            cluster_num = cluster_id.replace('cluster_', '')
            report.append(f"\nCluster {cluster_num}:")
            report.append(f"  Size: {stats['size']} runners ({stats['percentage']:.1f}%)")
            report.append(f"  Mean Finish Time: {stats['mean_finish_time']/60:.1f} minutes")
            report.append(f"  Mean Age: {stats['mean_age']:.1f} years")
            report.append(f"  Club Membership Rate: {stats['club_membership_rate']:.1%}")
            report.append(f"  Performance Percentile: {stats['mean_performance_percentile']:.1f}")
            
            if 'gender_distribution' in stats:
                report.append(f"  Gender Distribution: {stats['gender_distribution']}")
    
    # Cluster interpretations
    if 'kmeans_interpretations' in results['analysis']:
        report.append("\nCLUSTER INTERPRETATIONS")
        report.append("-" * 25)
        
        for cluster_id, interp in results['analysis']['kmeans_interpretations'].items():
            cluster_num = cluster_id.replace('cluster_', '')
            report.append(f"\nCluster {cluster_num}: {interp['label']}")
            report.append(f"  Performance Level: {interp['performance_level']}")
            report.append(f"  Age Group: {interp['age_group']}")
            if interp['key_characteristics']:
                report.append(f"  Key Characteristics: {', '.join(interp['key_characteristics'])}")
    
    # DBSCAN results
    report.append("\nDBSCAN CLUSTERING RESULTS")
    report.append("-" * 28)
    report.append(f"Number of Clusters: {results['dbscan']['n_clusters']}")
    report.append(f"Noise Points: {results['dbscan']['n_noise']} ({results['dbscan']['noise_ratio']:.1%})")
    
    if results['dbscan']['silhouette_score'] is not None:
        report.append(f"Silhouette Score: {results['dbscan']['silhouette_score']:.3f}")
    
    report.append(f"Best Parameters: {results['dbscan']['best_params']}")
    report.append("")
    
    # PCA results
    if 'pca' in results:
        report.append("PRINCIPAL COMPONENT ANALYSIS")
        report.append("-" * 32)
        report.append(f"Components for 95% variance: {results['pca']['n_components_95']}")
        report.append(f"First 3 components explain: {sum(results['pca']['explained_variance_ratio'][:3])*100:.1f}% of variance")
        
        report.append("\nTop 5 Principal Components:")
        for i in range(min(5, len(results['pca']['explained_variance_ratio']))):
            var_ratio = results['pca']['explained_variance_ratio'][i]
            report.append(f"  PC{i+1}: {var_ratio:.3f} ({var_ratio*100:.1f}%)")
        report.append("")
    
    # Model comparison
    report.append("MODEL COMPARISON")
    report.append("-" * 20)
    comparison = results['analysis']['model_comparison']
    
    for model, metrics in comparison.items():
        report.append(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                report.append(f"  {metric}: {value:.3f}")
            else:
                report.append(f"  {metric}: {value}")
    
    # Recommendations
    report.append("\nKEY FINDINGS & RECOMMENDATIONS")
    report.append("-" * 35)
    
    # Generate findings based on results
    silhouette_score = results['kmeans']['silhouette_score']
    if silhouette_score > 0.5:
        report.append("• Excellent cluster separation - clear runner segments identified")
    elif silhouette_score > 0.3:
        report.append("• Good cluster separation - meaningful runner groups found")
    else:
        report.append("• Moderate cluster separation - groups may overlap significantly")
    
    optimal_k = results['kmeans']['optimal_k']
    if optimal_k <= 4:
        report.append(f"• {optimal_k} distinct runner archetypes identified - manageable for targeted programs")
    else:
        report.append(f"• {optimal_k} runner segments - may benefit from grouping similar clusters")
    
    if 'pca' in results:
        n_components_95 = results['pca']['n_components_95']
        if n_components_95 <= 5:
            report.append("• Low dimensionality - performance determined by few key factors")
        else:
            report.append("• High dimensionality - complex performance relationships")
    
    report.append("• Use cluster assignments for personalized training recommendations")
    report.append("• Target marketing and event communication by cluster characteristics")
    report.append("• Consider cluster-specific performance targets and goals")
    
    # DBSCAN insights
    noise_ratio = results['dbscan']['noise_ratio']
    if noise_ratio < 0.05:
        report.append("• Low noise in DBSCAN - most runners fit clear patterns")
    elif noise_ratio > 0.20:
        report.append("• High noise in DBSCAN - many unique/outlier performance profiles")
    
    # Save report
    with open(output_dir / 'clustering_report.txt', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Main visualization function."""
    print("Creating clustering analysis visualizations...")
    
    # Setup output directory
    output_dir = ensure_output_dir('clustering')
    
    # Load results
    df, results, models = load_results()
    
    # Create visualizations
    print("Plotting cluster visualizations...")
    plot_cluster_visualization(df, results, output_dir)
    
    print("Plotting cluster analysis...")
    plot_cluster_analysis(df, results, output_dir)
    
    print("Creating 3D cluster visualization...")
    plot_3d_cluster_visualization(df, results, output_dir)
    
    print("Plotting PCA analysis...")
    plot_pca_analysis(results, output_dir)
    
    print("Creating clustering dashboard...")
    plot_clustering_dashboard(df, results, output_dir)
    
    print("Generating analysis report...")
    generate_clustering_report(df, results, output_dir)
    
    print(f"All visualizations saved to {output_dir}/")
    print("Generated files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()