#!/usr/bin/env python3
"""
Clustering Analysis for Race Performance Data
Segments runners into performance groups using K-means and DBSCAN clustering.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/clustering']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare for clustering analysis."""
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
    
    # Main performance measure
    df['finish_time'] = df['chip_time_seconds'].fillna(df['gun_time_seconds'])
    
    # Extract age from category
    def extract_age(category):
        if pd.isna(category):
            return np.nan
        try:
            age_part = ''.join(filter(str.isdigit, str(category)))
            return int(age_part) if age_part else np.nan
        except:
            return np.nan
    
    df['age'] = df['category'].apply(extract_age)
    
    # Gender encoding
    df['gender'] = df['gender'].str.upper()
    df['is_male'] = (df['gender'] == 'MALE').astype(int)
    
    # Club features
    df['has_club'] = (~df['club'].isna() & (df['club'] != 'None') & (df['club'] != '')).astype(int)
    
    # Calculate performance metrics
    if '10km_seconds' in df.columns:
        df['pace_10km'] = df['10km_seconds'] / 10000  # seconds per meter
        df['pace_overall'] = df['finish_time'] / 21097  # seconds per meter (half marathon)
        df['pace_consistency'] = df['pace_overall'] / df['pace_10km']  # pace degradation ratio
        df['negative_split'] = (df['pace_consistency'] < 1).astype(int)  # managed negative split
    
    # Performance percentiles
    df['performance_percentile'] = df['finish_time'].rank(pct=True) * 100
    
    # Age-graded performance (simplified)
    # Using basic age-grading factors (approximate)
    age_factors = {
        range(20, 30): 1.0,
        range(30, 35): 1.02,
        range(35, 40): 1.05,
        range(40, 45): 1.10,
        range(45, 50): 1.16,
        range(50, 55): 1.24,
        range(55, 60): 1.34,
        range(60, 65): 1.46,
        range(65, 100): 1.60
    }
    
    def get_age_factor(age):
        if pd.isna(age):
            return 1.0
        for age_range, factor in age_factors.items():
            if age in age_range:
                return factor
        return 1.6  # Default for very old
    
    df['age_factor'] = df['age'].apply(get_age_factor)
    df['age_graded_time'] = df['finish_time'] / df['age_factor']
    
    # Gender-adjusted performance
    # Approximate gender adjustment factor
    gender_factor = {'MALE': 1.0, 'FEMALE': 1.12}  # Women's times adjusted to male equivalent
    df['gender_factor'] = df['gender'].map(gender_factor).fillna(1.0)
    df['gender_adjusted_time'] = df['finish_time'] / df['gender_factor']
    
    return df

def prepare_clustering_features(df):
    """Prepare feature matrix for clustering."""
    
    # Define clustering features
    clustering_features = []
    
    # Core performance features
    core_features = ['finish_time', 'age', 'performance_percentile']
    clustering_features.extend(core_features)
    
    # Add pace features if available
    if '10km_seconds' in df.columns:
        pace_features = ['10km_seconds', 'pace_10km', 'pace_consistency']
        clustering_features.extend(pace_features)
    
    # Demographic features
    demographic_features = ['is_male', 'has_club']
    clustering_features.extend(demographic_features)
    
    # Advanced features
    advanced_features = ['age_graded_time', 'gender_adjusted_time']
    clustering_features.extend(advanced_features)
    
    # Create feature matrix
    feature_df = df[clustering_features].copy()
    
    # Remove rows with missing critical data
    complete_cases = feature_df['finish_time'].notna() & feature_df['age'].notna()
    feature_df_clean = feature_df[complete_cases]
    
    # Fill remaining missing values with median/mode
    for col in feature_df_clean.columns:
        if feature_df_clean[col].dtype in ['float64', 'int64']:
            feature_df_clean[col] = feature_df_clean[col].fillna(feature_df_clean[col].median())
        else:
            feature_df_clean[col] = feature_df_clean[col].fillna(feature_df_clean[col].mode()[0])
    
    return feature_df_clean, complete_cases

def perform_kmeans_clustering(X, X_scaled):
    """Perform K-means clustering with optimal k selection."""
    
    # Determine optimal number of clusters using multiple methods
    k_range = range(2, 11)
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    print("Finding optimal number of clusters...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for metrics
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
        else:
            silhouette_scores.append(-1)
            calinski_scores.append(0)
            davies_bouldin_scores.append(float('inf'))
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Fit final K-means model
    print(f"Fitting K-means with k={optimal_k}...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = final_kmeans.fit_predict(X_scaled)
    
    kmeans_results = {
        'optimal_k': optimal_k,
        'labels': kmeans_labels,
        'cluster_centers': final_kmeans.cluster_centers_,
        'inertia': final_kmeans.inertia_,
        'silhouette_score': silhouette_score(X_scaled, kmeans_labels),
        'calinski_harabasz_score': calinski_harabasz_score(X_scaled, kmeans_labels),
        'davies_bouldin_score': davies_bouldin_score(X_scaled, kmeans_labels),
        'k_selection_metrics': {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        },
        'model': final_kmeans
    }
    
    return kmeans_results

def perform_dbscan_clustering(X, X_scaled):
    """Perform DBSCAN clustering with parameter optimization."""
    
    # Try different eps values
    eps_range = np.arange(0.3, 2.0, 0.1)
    min_samples_range = [5, 10, 15, 20]
    
    best_score = -1
    best_params = None
    best_labels = None
    
    print("Optimizing DBSCAN parameters...")
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            # Calculate metrics (if we have more than 1 cluster and not all noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1 and n_noise < len(labels) * 0.9:  # Not too much noise
                try:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_labels = labels
                except:
                    continue
    
    # Fit final DBSCAN model
    if best_params is not None:
        print(f"Fitting DBSCAN with eps={best_params['eps']:.2f}, min_samples={best_params['min_samples']}")
        final_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
        dbscan_labels = final_dbscan.fit_predict(X_scaled)
    else:
        # Fallback parameters
        print("Using fallback DBSCAN parameters...")
        final_dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan_labels = final_dbscan.fit_predict(X_scaled)
        best_params = {'eps': 0.5, 'min_samples': 10}
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    dbscan_results = {
        'best_params': best_params,
        'labels': dbscan_labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(dbscan_labels),
        'silhouette_score': silhouette_score(X_scaled, dbscan_labels) if n_clusters > 1 else None,
        'model': final_dbscan
    }
    
    return dbscan_results

def analyze_clusters(X, kmeans_results, dbscan_results, df_original, complete_cases):
    """Analyze cluster characteristics and performance patterns."""
    
    analysis = {}
    
    # Prepare dataframe with cluster labels
    cluster_df = X.copy()
    cluster_df['kmeans_cluster'] = kmeans_results['labels']
    cluster_df['dbscan_cluster'] = dbscan_results['labels']
    
    # Add original data for interpretation
    original_subset = df_original[complete_cases].reset_index(drop=True)
    cluster_df = pd.concat([cluster_df, original_subset[['gender', 'club', 'category']]], axis=1)
    
    # K-means cluster analysis
    print("Analyzing K-means clusters...")
    kmeans_analysis = {}
    
    for cluster_id in range(kmeans_results['optimal_k']):
        cluster_mask = cluster_df['kmeans_cluster'] == cluster_id
        cluster_data = cluster_df[cluster_mask]
        
        cluster_stats = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(cluster_df) * 100,
            'mean_finish_time': cluster_data['finish_time'].mean(),
            'median_finish_time': cluster_data['finish_time'].median(),
            'std_finish_time': cluster_data['finish_time'].std(),
            'mean_age': cluster_data['age'].mean(),
            'gender_distribution': cluster_data['gender'].value_counts().to_dict(),
            'club_membership_rate': cluster_data['has_club'].mean(),
            'mean_performance_percentile': cluster_data['performance_percentile'].mean()
        }
        
        # Add pace analysis if available
        if 'pace_consistency' in cluster_data.columns:
            cluster_stats.update({
                'mean_pace_consistency': cluster_data['pace_consistency'].mean(),
                'negative_split_rate': cluster_data['negative_split'].mean() if 'negative_split' in cluster_data.columns else None
            })
        
        kmeans_analysis[f'cluster_{cluster_id}'] = cluster_stats
    
    analysis['kmeans_clusters'] = kmeans_analysis
    
    # DBSCAN cluster analysis
    print("Analyzing DBSCAN clusters...")
    dbscan_analysis = {}
    
    unique_labels = set(dbscan_results['labels'])
    for cluster_id in unique_labels:
        if cluster_id == -1:
            label = 'noise'
        else:
            label = f'cluster_{cluster_id}'
        
        cluster_mask = cluster_df['dbscan_cluster'] == cluster_id
        cluster_data = cluster_df[cluster_mask]
        
        cluster_stats = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(cluster_df) * 100,
            'mean_finish_time': cluster_data['finish_time'].mean(),
            'median_finish_time': cluster_data['finish_time'].median(),
            'std_finish_time': cluster_data['finish_time'].std(),
            'mean_age': cluster_data['age'].mean(),
            'gender_distribution': cluster_data['gender'].value_counts().to_dict(),
            'club_membership_rate': cluster_data['has_club'].mean(),
            'mean_performance_percentile': cluster_data['performance_percentile'].mean()
        }
        
        dbscan_analysis[label] = cluster_stats
    
    analysis['dbscan_clusters'] = dbscan_analysis
    
    # Cluster interpretation
    print("Generating cluster interpretations...")
    
    # K-means interpretations
    kmeans_interpretations = {}
    for cluster_id in range(kmeans_results['optimal_k']):
        stats = kmeans_analysis[f'cluster_{cluster_id}']
        
        # Classify cluster based on performance percentile
        perf_percentile = stats['mean_performance_percentile']
        if perf_percentile <= 25:
            performance_level = "Elite"
        elif perf_percentile <= 50:
            performance_level = "Competitive"
        elif perf_percentile <= 75:
            performance_level = "Recreational"
        else:
            performance_level = "Casual"
        
        # Age classification
        mean_age = stats['mean_age']
        if mean_age < 35:
            age_group = "Young"
        elif mean_age < 50:
            age_group = "Middle-aged"
        else:
            age_group = "Veteran"
        
        interpretation = f"{performance_level} {age_group} Runners"
        
        # Add specific characteristics
        characteristics = []
        if stats['club_membership_rate'] > 0.7:
            characteristics.append("High club membership")
        if stats['gender_distribution'].get('MALE', 0) > stats['gender_distribution'].get('FEMALE', 0) * 1.5:
            characteristics.append("Male-dominated")
        elif stats['gender_distribution'].get('FEMALE', 0) > stats['gender_distribution'].get('MALE', 0) * 1.5:
            characteristics.append("Female-dominated")
        
        if characteristics:
            interpretation += f" ({', '.join(characteristics)})"
        
        kmeans_interpretations[f'cluster_{cluster_id}'] = {
            'label': interpretation,
            'key_characteristics': characteristics,
            'performance_level': performance_level,
            'age_group': age_group
        }
    
    analysis['kmeans_interpretations'] = kmeans_interpretations
    
    # Model comparison
    analysis['model_comparison'] = {
        'kmeans': {
            'n_clusters': kmeans_results['optimal_k'],
            'silhouette_score': kmeans_results['silhouette_score'],
            'method': 'Partitional clustering'
        },
        'dbscan': {
            'n_clusters': dbscan_results['n_clusters'],
            'silhouette_score': dbscan_results['silhouette_score'],
            'noise_ratio': dbscan_results['noise_ratio'],
            'method': 'Density-based clustering'
        }
    }
    
    return analysis, cluster_df

def perform_pca_analysis(X_scaled):
    """Perform PCA for dimensionality reduction and visualization."""
    
    print("Performing PCA analysis...")
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    pca_results = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'n_components_95': n_components_95,
        'transformed_data': X_pca[:, :3].tolist(),  # First 3 components for visualization
        'feature_loadings': pca.components_[:3].tolist(),  # First 3 component loadings
        'model': pca
    }
    
    return pca_results

def save_results(X, kmeans_results, dbscan_results, pca_results, analysis, cluster_df):
    """Save all clustering analysis results."""
    
    # Save cluster assignments with original data
    cluster_df.to_csv('data/processed/clustering_data.csv', index=False)
    
    # Prepare results for JSON serialization
    results_to_save = {
        'kmeans': {k: v for k, v in kmeans_results.items() if k != 'model'},
        'dbscan': {k: v for k, v in dbscan_results.items() if k != 'model'},
        'pca': {k: v for k, v in pca_results.items() if k != 'model'},
        'analysis': analysis,
        'data_summary': {
            'total_samples': len(X),
            'features_used': X.columns.tolist(),
            'feature_stats': X.describe().to_dict()
        }
    }
    
    # Save results to JSON
    with open('data/processed/clustering_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    # Save model objects
    models = {
        'kmeans': kmeans_results['model'],
        'dbscan': dbscan_results['model'],
        'pca': pca_results['model'],
        'scaler': StandardScaler().fit(X)  # Save scaler for future use
    }
    
    with open('data/processed/clustering_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Results saved to data/processed/")
    
    # Print summary
    print(f"\nClustering Analysis Summary:")
    print(f"Total samples clustered: {len(X)}")
    print(f"K-means clusters: {kmeans_results['optimal_k']}")
    print(f"K-means silhouette score: {kmeans_results['silhouette_score']:.3f}")
    print(f"DBSCAN clusters: {dbscan_results['n_clusters']}")
    print(f"DBSCAN noise ratio: {dbscan_results['noise_ratio']:.1%}")
    
    if dbscan_results['silhouette_score']:
        print(f"DBSCAN silhouette score: {dbscan_results['silhouette_score']:.3f}")
    
    print(f"PCA components for 95% variance: {pca_results['n_components_95']}")

def main():
    """Main execution function."""
    print("Starting Clustering Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Prepare features
    print("Preparing clustering features...")
    X, complete_cases = prepare_clustering_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans_results = perform_kmeans_clustering(X, X_scaled)
    dbscan_results = perform_dbscan_clustering(X, X_scaled)
    
    # PCA analysis
    pca_results = perform_pca_analysis(X_scaled)
    
    # Analyze clusters
    print("Analyzing cluster characteristics...")
    analysis, cluster_df = analyze_clusters(X, kmeans_results, dbscan_results, df, complete_cases)
    
    # Save results
    save_results(X, kmeans_results, dbscan_results, pca_results, analysis, cluster_df)
    
    print("Clustering analysis complete!")

if __name__ == "__main__":
    main()