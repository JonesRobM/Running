#!/usr/bin/env python3
"""
Time Series Analysis for Race Performance Data
Analyzes pacing strategies and temporal performance patterns using time series techniques.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/time_series']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare for time series analysis."""
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
    
    # Extract age and other features
    def extract_age(category):
        if pd.isna(category):
            return np.nan
        try:
            age_part = ''.join(filter(str.isdigit, str(category)))
            return int(age_part) if age_part else np.nan
        except:
            return np.nan
    
    df['age'] = df['category'].apply(extract_age)
    df['gender'] = df['gender'].str.upper()
    
    # Calculate pacing metrics if 10km split available
    if '10km_seconds' in df.columns and df['10km_seconds'].notna().sum() > 100:
        df['pace_10km'] = df['10km_seconds'] / 10000  # seconds per meter
        df['pace_second_half'] = (df['finish_time'] - df['10km_seconds']) / 11097  # seconds per meter
        df['pace_overall'] = df['finish_time'] / 21097  # seconds per meter
        
        # Pacing ratios and patterns
        df['negative_split'] = (df['pace_10km'] > df['pace_second_half']).astype(int)
        df['pace_ratio'] = df['pace_second_half'] / df['pace_10km']
        df['pace_differential'] = df['pace_second_half'] - df['pace_10km']
        
        # Create pacing categories
        df['pacing_strategy'] = pd.cut(df['pace_ratio'], 
                                     bins=[0, 0.95, 1.05, 1.15, np.inf],
                                     labels=['Negative Split', 'Even Pace', 'Moderate Positive', 'Severe Positive'])
    
    # Performance percentiles for time series analysis
    df['performance_rank'] = df['finish_time'].rank()
    df['performance_percentile'] = df['performance_rank'] / len(df) * 100
    
    # Create time-based features (if position data available)
    if 'position' in df.columns:
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
        df['position_change'] = df['position_numeric'] - df['performance_rank']
    
    return df

def analyze_pacing_patterns(df):
    """Analyze pacing strategies using time series techniques."""
    results = {}
    
    print("Analyzing pacing patterns...")
    
    # Check if we have split data
    if '10km_seconds' not in df.columns or df['10km_seconds'].notna().sum() < 50:
        print("Insufficient split time data for detailed pacing analysis")
        return {'error': 'Insufficient split time data'}
    
    # Basic pacing statistics
    pacing_data = df[df['10km_seconds'].notna()].copy()
    
    # 1. Pacing strategy distribution
    pacing_distribution = pacing_data['pacing_strategy'].value_counts().to_dict()
    results['pacing_distribution'] = pacing_distribution
    
    # 2. Negative split analysis
    negative_split_rate = pacing_data['negative_split'].mean()
    results['negative_split_rate'] = negative_split_rate
    
    # 3. Pace ratio statistics
    pace_ratio_stats = {
        'mean': float(pacing_data['pace_ratio'].mean()),
        'median': float(pacing_data['pace_ratio'].median()),
        'std': float(pacing_data['pace_ratio'].std()),
        'q25': float(pacing_data['pace_ratio'].quantile(0.25)),
        'q75': float(pacing_data['pace_ratio'].quantile(0.75))
    }
    results['pace_ratio_stats'] = pace_ratio_stats
    
    # 4. Performance by pacing strategy
    performance_by_pacing = {}
    for strategy in pacing_data['pacing_strategy'].unique():
        if pd.notna(strategy):
            strategy_data = pacing_data[pacing_data['pacing_strategy'] == strategy]
            performance_by_pacing[strategy] = {
                'count': len(strategy_data),
                'mean_time': float(strategy_data['finish_time'].mean()),
                'median_time': float(strategy_data['finish_time'].median()),
                'mean_percentile': float(strategy_data['performance_percentile'].mean()),
                'completion_rate': float(strategy_data['finish_time'].notna().mean())
            }
    
    results['performance_by_pacing'] = performance_by_pacing
    
    # 5. Demographic pacing analysis
    demographic_pacing = {}
    
    # By gender
    if 'gender' in pacing_data.columns:
        gender_pacing = pacing_data.groupby('gender').agg({
            'pace_ratio': ['mean', 'std'],
            'negative_split': 'mean',
            'pace_differential': 'mean'
        }).round(4)
        demographic_pacing['gender'] = gender_pacing.to_dict()
    
    # By age group
    if 'age' in pacing_data.columns:
        pacing_data['age_group'] = pd.cut(pacing_data['age'], 
                                        bins=[0, 30, 40, 50, 60, 100],
                                        labels=['<30', '30-39', '40-49', '50-59', '60+'])
        
        age_pacing = pacing_data.groupby('age_group').agg({
            'pace_ratio': ['mean', 'std'],
            'negative_split': 'mean',
            'pace_differential': 'mean'
        }).round(4)
        demographic_pacing['age_group'] = age_pacing.to_dict()
    
    results['demographic_pacing'] = demographic_pacing
    
    return results

def analyze_performance_trends(df):
    """Analyze performance trends and patterns over the race field."""
    results = {}
    
    print("Analyzing performance trends...")
    
    # 1. Performance distribution analysis
    performance_stats = {
        'mean_time': float(df['finish_time'].mean()),
        'median_time': float(df['finish_time'].median()),
        'std_time': float(df['finish_time'].std()),
        'skewness': float(stats.skew(df['finish_time'].dropna())),
        'kurtosis': float(stats.kurtosis(df['finish_time'].dropna()))
    }
    results['performance_distribution'] = performance_stats
    
    # 2. Quartile analysis with time series approach
    quartiles = df['finish_time'].quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
    results['performance_quartiles'] = {f"q{int(k*100)}": v for k, v in quartiles.items()}
    
    # 3. Performance gaps analysis
    sorted_times = df['finish_time'].dropna().sort_values()
    
    # Calculate gaps between consecutive finishers
    time_gaps = sorted_times.diff().dropna()
    
    gap_analysis = {
        'mean_gap': float(time_gaps.mean()),
        'median_gap': float(time_gaps.median()),
        'max_gap': float(time_gaps.max()),
        'std_gap': float(time_gaps.std())
    }
    
    # Find significant gaps (outliers)
    gap_threshold = time_gaps.mean() + 2 * time_gaps.std()
    significant_gaps = time_gaps[time_gaps > gap_threshold]
    
    gap_analysis['significant_gaps'] = {
        'count': len(significant_gaps),
        'positions': significant_gaps.index.tolist(),
        'threshold': float(gap_threshold)
    }
    
    results['gap_analysis'] = gap_analysis
    
    # 4. Performance bands analysis
    # Identify natural performance bands using change point detection
    time_changes = np.diff(sorted_times.values)
    
    # Simple change point detection using rolling statistics
    window_size = min(50, len(time_changes) // 10)
    if window_size > 5:
        rolling_mean = pd.Series(time_changes).rolling(window_size).mean()
        rolling_std = pd.Series(time_changes).rolling(window_size).std()
        
        # Identify points where rate of change significantly increases
        change_points = []
        for i in range(window_size, len(time_changes) - window_size):
            if (time_changes[i] > rolling_mean.iloc[i] + 2 * rolling_std.iloc[i] and
                rolling_std.iloc[i] > 0):
                change_points.append(i)
        
        results['change_points'] = {
            'positions': change_points,
            'times': [float(sorted_times.iloc[i]) for i in change_points],
            'interpretation': 'Positions where performance gaps significantly increase'
        }
    
    # 5. Age-performance time series
    if 'age' in df.columns:
        age_performance = df.groupby('age')['finish_time'].agg(['count', 'mean', 'std']).reset_index()
        age_performance = age_performance[age_performance['count'] >= 3]  # Only ages with 3+ participants
        
        # Smooth age-performance curve
        age_trend = {
            'ages': age_performance['age'].tolist(),
            'mean_times': age_performance['mean'].tolist(),
            'participant_counts': age_performance['count'].tolist()
        }
        
        # Calculate age-graded performance trends
        if len(age_performance) > 5:
            # Fit polynomial trend
            age_values = age_performance['age'].values
            time_values = age_performance['mean'].values
            
            # Fit quadratic polynomial (typical age-performance relationship)
            coeffs = np.polyfit(age_values, time_values, 2)
            age_trend['polynomial_coeffs'] = coeffs.tolist()
            
            # Find optimal age (minimum of parabola if opening upward)
            if coeffs[0] > 0:  # Parabola opens upward
                optimal_age = -coeffs[1] / (2 * coeffs[0])
                if 20 <= optimal_age <= 80:  # Reasonable age range
                    age_trend['optimal_age'] = float(optimal_age)
        
        results['age_performance_trend'] = age_trend
    
    return results

def analyze_outliers_and_anomalies(df):
    """Detect performance outliers and anomalies using time series methods."""
    results = {}
    
    print("Analyzing outliers and anomalies...")
    
    # 1. Statistical outliers
    times = df['finish_time'].dropna()
    
    # Z-score method
    z_scores = np.abs(stats.zscore(times))
    z_outliers = df[z_scores > 3].copy()
    
    # IQR method
    Q1 = times.quantile(0.25)
    Q3 = times.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    iqr_outliers = df[(df['finish_time'] < lower_bound) | (df['finish_time'] > upper_bound)].copy()
    
    outlier_analysis = {
        'z_score_outliers': {
            'count': len(z_outliers),
            'percentage': len(z_outliers) / len(df) * 100,
            'time_range': [float(z_outliers['finish_time'].min()), float(z_outliers['finish_time'].max())] if len(z_outliers) > 0 else []
        },
        'iqr_outliers': {
            'count': len(iqr_outliers),
            'percentage': len(iqr_outliers) / len(df) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    }
    
    results['outlier_analysis'] = outlier_analysis
    
    # 2. Pacing anomalies (if split data available)
    if '10km_seconds' in df.columns:
        pacing_data = df[df['10km_seconds'].notna()].copy()
        
        if len(pacing_data) > 50:
            # Extreme pacing ratios
            pace_outliers = pacing_data[
                (pacing_data['pace_ratio'] < 0.8) |  # Very negative split
                (pacing_data['pace_ratio'] > 1.5)    # Very positive split
            ].copy()
            
            pacing_anomalies = {
                'extreme_pacing_count': len(pace_outliers),
                'extreme_negative_splits': len(pacing_data[pacing_data['pace_ratio'] < 0.8]),
                'extreme_positive_splits': len(pacing_data[pacing_data['pace_ratio'] > 1.5]),
                'most_extreme_negative': float(pacing_data['pace_ratio'].min()),
                'most_extreme_positive': float(pacing_data['pace_ratio'].max())
            }
            
            results['pacing_anomalies'] = pacing_anomalies
    
    # 3. Age-performance anomalies
    if 'age' in df.columns:
        age_data = df[df['age'].notna()].copy()
        
        # Find unusually fast/slow performers for their age
        age_groups = age_data.groupby('age')['finish_time'].agg(['mean', 'std']).reset_index()
        age_groups = age_groups[age_groups['std'].notna()]
        
        age_anomalies = []
        for _, person in age_data.iterrows():
            age_stats = age_groups[age_groups['age'] == person['age']]
            if len(age_stats) > 0:
                expected_time = age_stats['mean'].iloc[0]
                age_std = age_stats['std'].iloc[0]
                
                if pd.notna(age_std) and age_std > 0:
                    z_score = (person['finish_time'] - expected_time) / age_std
                    if abs(z_score) > 2:  # More than 2 standard deviations
                        age_anomalies.append({
                            'age': person['age'],
                            'actual_time': person['finish_time'],
                            'expected_time': expected_time,
                            'z_score': z_score,
                            'type': 'exceptionally_fast' if z_score < -2 else 'exceptionally_slow'
                        })
        
        results['age_performance_anomalies'] = {
            'count': len(age_anomalies),
            'exceptionally_fast': len([a for a in age_anomalies if a['type'] == 'exceptionally_fast']),
            'exceptionally_slow': len([a for a in age_anomalies if a['type'] == 'exceptionally_slow'])
        }
    
    return results

def perform_time_series_decomposition(df):
    """Perform time series decomposition on performance data."""
    results = {}
    
    print("Performing time series decomposition...")
    
    # Create a position-based time series
    sorted_df = df.sort_values('finish_time').reset_index(drop=True)
    sorted_df['position'] = range(1, len(sorted_df) + 1)
    
    # 1. Trend analysis
    positions = sorted_df['position'].values
    times = sorted_df['finish_time'].values
    
    # Fit polynomial trends of different orders
    trend_analysis = {}
    
    for order in [1, 2, 3]:
        coeffs = np.polyfit(positions, times, order)
        fitted_values = np.polyval(coeffs, positions)
        
        # Calculate R-squared
        ss_res = np.sum((times - fitted_values) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        trend_analysis[f'order_{order}'] = {
            'coefficients': coeffs.tolist(),
            'r_squared': float(r_squared),
            'rmse': float(np.sqrt(np.mean((times - fitted_values) ** 2)))
        }
    
    results['trend_analysis'] = trend_analysis
    
    # 2. Residual analysis
    # Use best fitting trend (highest R-squared)
    best_order = max(trend_analysis.keys(), key=lambda x: trend_analysis[x]['r_squared'])
    best_coeffs = trend_analysis[best_order]['coefficients']
    trend_component = np.polyval(best_coeffs, positions)
    residuals = times - trend_component
    
    residual_analysis = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'autocorrelation_lag1': float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]),
        'runs_test_p_value': None  # Would need scipy.stats for proper runs test
    }
    
    # Simple runs test approximation
    runs = 1
    for i in range(1, len(residuals)):
        if (residuals[i] > 0) != (residuals[i-1] > 0):
            runs += 1
    
    expected_runs = 2 * np.sum(residuals > 0) * np.sum(residuals <= 0) / len(residuals) + 1
    residual_analysis['runs_count'] = runs
    residual_analysis['expected_runs'] = float(expected_runs)
    
    results['residual_analysis'] = residual_analysis
    
    # 3. Seasonality/cyclical patterns (in performance data)
    # Look for periodic patterns in residuals
    if len(residuals) > 100:
        # Simple frequency analysis
        fft = np.fft.fft(residuals)
        frequencies = np.fft.fftfreq(len(residuals))
        power_spectrum = np.abs(fft) ** 2
        
        # Find dominant frequencies
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_frequency = frequencies[dominant_freq_idx]
        dominant_period = 1 / abs(dominant_frequency) if dominant_frequency != 0 else None
        
        frequency_analysis = {
            'dominant_frequency': float(dominant_frequency),
            'dominant_period': float(dominant_period) if dominant_period else None,
            'power_spectrum_peak': float(power_spectrum[dominant_freq_idx])
        }
        
        results['frequency_analysis'] = frequency_analysis
    
    return results

def save_results(df, pacing_results, trend_results, outlier_results, decomposition_results):
    """Save all analysis results."""
    
    # Save processed dataframe
    df.to_csv('data/processed/time_series_data.csv', index=False)
    
    # Combine all results
    full_results = {
        'pacing_analysis': pacing_results,
        'performance_trends': trend_results,
        'outlier_analysis': outlier_results,
        'time_series_decomposition': decomposition_results,
        'data_summary': {
            'total_participants': len(df),
            'with_split_times': int(df['10km_seconds'].notna().sum()) if '10km_seconds' in df.columns else 0,
            'age_range': [float(df['age'].min()), float(df['age'].max())] if 'age' in df.columns else [],
            'time_range_minutes': [float(df['finish_time'].min()/60), float(df['finish_time'].max()/60)],
            'negative_split_rate': float(df['negative_split'].mean()) if 'negative_split' in df.columns else None
        }
    }
    
    # Save results to JSON
    with open('data/processed/time_series_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save any model objects (if created)
    models = {}
    
    with open('data/processed/time_series_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Results saved to data/processed/")
    
    # Print summary
    print(f"\nTime Series Analysis Summary:")
    print(f"Total participants: {len(df)}")
    
    if 'error' not in pacing_results:
        print(f"Participants with split times: {full_results['data_summary']['with_split_times']}")
        print(f"Negative split rate: {full_results['data_summary']['negative_split_rate']:.1%}")
    
    print(f"Performance range: {full_results['data_summary']['time_range_minutes'][0]:.1f} - {full_results['data_summary']['time_range_minutes'][1]:.1f} minutes")
    
    if 'outlier_analysis' in full_results:
        outlier_pct = full_results['outlier_analysis']['outlier_analysis']['iqr_outliers']['percentage']
        print(f"Performance outliers: {outlier_pct:.1f}%")

def main():
    """Main execution function."""
    print("Starting Time Series Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Perform analyses
    pacing_results = analyze_pacing_patterns(df)
    trend_results = analyze_performance_trends(df)
    outlier_results = analyze_outliers_and_anomalies(df)
    decomposition_results = perform_time_series_decomposition(df)
    
    # Save results
    save_results(df, pacing_results, trend_results, outlier_results, decomposition_results)
    
    print("Time series analysis complete!")

if __name__ == "__main__":
    main()