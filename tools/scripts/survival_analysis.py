#!/usr/bin/env python3
"""
Survival Analysis for Race Performance Data
Analyzes completion times and factors affecting performance using survival analysis techniques.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/survival_analysis']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare for survival analysis."""
    # Load the scraped race data
    df = pd.read_csv('data/raw/race_results.csv')
    
    # Convert time strings to seconds for analysis
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
    
    # Create event indicator (1 = finished, 0 = DNF)
    df['event'] = (~df['chip_time_seconds'].isna()).astype(int)
    
    # Create duration variable (use chip time, fallback to gun time)
    df['duration'] = df['chip_time_seconds'].fillna(df['gun_time_seconds'])
    
    # Age categories to numeric (extract age from category)
    def extract_age(category):
        if pd.isna(category):
            return np.nan
        try:
            # Handle formats like 'M40', 'F35', etc.
            age_part = ''.join(filter(str.isdigit, str(category)))
            return int(age_part) if age_part else np.nan
        except:
            return np.nan
    
    df['age'] = df['category'].apply(extract_age)
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 30, 40, 50, 60, 100], 
                           labels=['<30', '30-39', '40-49', '50-59', '60+'])
    
    # Clean gender
    df['gender'] = df['gender'].str.upper()
    
    # Create club indicator
    df['has_club'] = (~df['club'].isna() & (df['club'] != 'None')).astype(int)
    
    # Remove rows without duration data
    df_clean = df.dropna(subset=['duration'])
    
    return df_clean

def perform_survival_analysis(df):
    """Perform comprehensive survival analysis."""
    results = {}
    
    # 1. Kaplan-Meier Analysis by Gender
    print("Performing Kaplan-Meier analysis...")
    kmf_results = {}
    
    for gender in ['MALE', 'FEMALE']:
        if gender in df['gender'].values:
            gender_data = df[df['gender'] == gender]
            kmf = KaplanMeierFitter()
            kmf.fit(gender_data['duration'], 
                   event_observed=gender_data['event'],
                   label=f'{gender}')
            
            kmf_results[gender] = {
                'survival_function': kmf.survival_function_,
                'median_survival': kmf.median_survival_time_,
                'confidence_interval': kmf.confidence_interval_
            }
    
    results['kaplan_meier'] = kmf_results
    
    # 2. Logrank test for gender differences
    male_data = df[df['gender'] == 'MALE']
    female_data = df[df['gender'] == 'FEMALE']
    
    if len(male_data) > 0 and len(female_data) > 0:
        logrank_result = logrank_test(
            male_data['duration'], female_data['duration'],
            male_data['event'], female_data['event']
        )
        
        results['logrank_test'] = {
            'test_statistic': logrank_result.test_statistic,
            'p_value': logrank_result.p_value,
            'null_hypothesis': 'No difference in survival curves between genders'
        }
    
    # 3. Cox Proportional Hazards Model
    print("Fitting Cox Proportional Hazards model...")
    
    # Prepare data for Cox regression
    cox_data = df.copy()
    
    # Create dummy variables
    cox_data = pd.get_dummies(cox_data, columns=['gender', 'age_group'], prefix=['gender', 'age'])
    
    # Select features for Cox model
    cox_features = [col for col in cox_data.columns if 
                   col.startswith('gender_') or 
                   col.startswith('age_') or 
                   col == 'has_club']
    
    # Remove rows with missing data
    cox_data_clean = cox_data[['duration', 'event'] + cox_features].dropna()
    
    if len(cox_data_clean) > 50:  # Ensure sufficient data
        cph = CoxPHFitter()
        cph.fit(cox_data_clean, duration_col='duration', event_col='event')
        
        results['cox_model'] = {
            'summary': cph.summary.to_dict(),
            'hazard_ratios': cph.hazard_ratios_.to_dict(),
            'confidence_intervals': cph.confidence_intervals_.to_dict(),
            'concordance_index': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_,
            'AIC': cph.AIC_
        }
        
        # Store the fitted model
        results['cox_model_object'] = cph
    
    # 4. Parametric survival analysis (Weibull)
    print("Fitting Weibull survival model...")
    wf = WeibullFitter()
    wf.fit(df['duration'], event_observed=df['event'])
    
    results['weibull_model'] = {
        'summary': wf.summary.to_dict(),
        'lambda_': wf.lambda_,
        'rho_': wf.rho_,
        'median_survival': wf.median_survival_time_,
        'survival_function': wf.survival_function_
    }
    
    # 5. Performance quartile analysis
    quartiles = df['duration'].quantile([0.25, 0.5, 0.75]).to_dict()
    df['performance_quartile'] = pd.qcut(df['duration'], 
                                       q=4, 
                                       labels=['Fast', 'Medium-Fast', 'Medium-Slow', 'Slow'])
    
    quartile_stats = df.groupby('performance_quartile').agg({
        'age': ['mean', 'std'],
        'has_club': 'mean',
        'duration': ['mean', 'std', 'count']
    }).round(2)
    
    results['quartile_analysis'] = {
        'quartiles': quartiles,
        'quartile_stats': quartile_stats.to_dict()
    }
    
    return results, df

def save_results(results, df):
    """Save analysis results and processed data."""
    
    # Save processed dataframe
    df.to_csv('data/processed/survival_analysis_data.csv', index=False)
    
    # Save results (excluding model objects)
    results_to_save = {k: v for k, v in results.items() 
                      if k not in ['cox_model_object']}
    
    with open('data/processed/survival_analysis_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    # Save model objects separately
    model_objects = {}
    if 'cox_model_object' in results:
        model_objects['cox_model'] = results['cox_model_object']
    
    with open('data/processed/survival_analysis_models.pkl', 'wb') as f:
        pickle.dump(model_objects, f)
    
    print("Results saved to data/processed/")
    
    # Print summary statistics
    print(f"\nSurvival Analysis Summary:")
    print(f"Total participants: {len(df)}")
    print(f"Event rate (finished): {df['event'].mean():.1%}")
    print(f"Median completion time: {df['duration'].median()/60:.1f} minutes")
    
    if 'cox_model' in results:
        print(f"Cox model concordance index: {results['cox_model']['concordance_index']:.3f}")
    
    if 'logrank_test' in results:
        print(f"Gender difference p-value: {results['logrank_test']['p_value']:.3f}")

def main():
    """Main execution function."""
    print("Starting Survival Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Perform analysis
    results, df_processed = perform_survival_analysis(df)
    
    # Save results
    save_results(results, df_processed)
    
    print("Survival analysis complete!")

if __name__ == "__main__":
    main()