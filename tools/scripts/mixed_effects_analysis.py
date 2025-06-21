#!/usr/bin/env python3
"""
Mixed Effects Models for Race Performance Data
Analyzes hierarchical effects (clubs, age groups) on race performance.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/mixed_effects']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare for mixed effects analysis."""
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
    
    # Use chip time as primary outcome, fallback to gun time
    df['finish_time'] = df['chip_time_seconds'].fillna(df['gun_time_seconds'])
    
    # Log transform for normality
    df['log_finish_time'] = np.log(df['finish_time'])
    
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
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 30, 40, 50, 60, 100], 
                           labels=['Under30', '30-39', '40-49', '50-59', '60Plus'])
    
    # Clean gender
    df['gender'] = df['gender'].str.upper()
    
    # Clean club data
    df['club_clean'] = df['club'].fillna('No Club')
    df['club_clean'] = df['club_clean'].replace(['None', 'none', ''], 'No Club')
    
    # Only keep clubs with at least 5 members for meaningful random effects
    club_counts = df['club_clean'].value_counts()
    valid_clubs = club_counts[club_counts >= 5].index
    df['club_analysis'] = df['club_clean'].apply(lambda x: x if x in valid_clubs else 'Other/Small Club')
    
    # Calculate pace variables
    if '10km_seconds' in df.columns:
        df['pace_10km'] = df['10km_seconds'] / 10000  # seconds per meter
        df['pace_overall'] = df['finish_time'] / 21097  # seconds per meter (half marathon distance)
        df['pace_ratio'] = df['pace_overall'] / df['pace_10km']  # pace degradation
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['finish_time', 'gender', 'age'])
    
    # Remove extreme outliers (beyond 3 standard deviations)
    z_scores = np.abs(stats.zscore(df_clean['finish_time']))
    df_clean = df_clean[z_scores < 3]
    
    return df_clean

def fit_mixed_effects_models(df):
    """Fit various mixed effects models."""
    models = {}
    results = {}
    
    print("Fitting mixed effects models...")
    
    # Model 1: Basic model with club as random effect
    print("  Model 1: Club random effects...")
    try:
        model1 = mixedlm("log_finish_time ~ age + C(gender)", 
                        df, groups=df["club_analysis"])
        fitted1 = model1.fit()
        models['club_random'] = fitted1
        results['club_random'] = {
            'summary': str(fitted1.summary()),
            'aic': fitted1.aic,
            'bic': fitted1.bic,
            'log_likelihood': fitted1.llf,
            'random_effects_var': fitted1.cov_re.iloc[0, 0] if hasattr(fitted1, 'cov_re') else None,
            'residual_var': fitted1.scale,
            'fixed_effects': fitted1.params.to_dict(),
            'fixed_effects_pvalues': fitted1.pvalues.to_dict()
        }
    except Exception as e:
        print(f"    Error fitting Model 1: {e}")
        results['club_random'] = {'error': str(e)}
    
    # Model 2: Age group as random effect
    print("  Model 2: Age group random effects...")
    try:
        model2 = mixedlm("log_finish_time ~ age + C(gender)", 
                        df, groups=df["age_group"])
        fitted2 = model2.fit()
        models['age_group_random'] = fitted2
        results['age_group_random'] = {
            'summary': str(fitted2.summary()),
            'aic': fitted2.aic,
            'bic': fitted2.bic,
            'log_likelihood': fitted2.llf,
            'random_effects_var': fitted2.cov_re.iloc[0, 0] if hasattr(fitted2, 'cov_re') else None,
            'residual_var': fitted2.scale,
            'fixed_effects': fitted2.params.to_dict(),
            'fixed_effects_pvalues': fitted2.pvalues.to_dict()
        }
    except Exception as e:
        print(f"    Error fitting Model 2: {e}")
        results['age_group_random'] = {'error': str(e)}
    
    # Model 3: Complex model with both club and age effects
    print("  Model 3: Complex hierarchical model...")
    try:
        # Create a combined grouping variable
        df['club_age_group'] = df['club_analysis'] + "_" + df['age_group'].astype(str)
        
        model3 = mixedlm("log_finish_time ~ age + C(gender) + pace_ratio", 
                        df[df['pace_ratio'].notna()], 
                        groups=df[df['pace_ratio'].notna()]["club_analysis"])
        fitted3 = model3.fit()
        models['complex_model'] = fitted3
        results['complex_model'] = {
            'summary': str(fitted3.summary()),
            'aic': fitted3.aic,
            'bic': fitted3.bic,
            'log_likelihood': fitted3.llf,
            'random_effects_var': fitted3.cov_re.iloc[0, 0] if hasattr(fitted3, 'cov_re') else None,
            'residual_var': fitted3.scale,
            'fixed_effects': fitted3.params.to_dict(),
            'fixed_effects_pvalues': fitted3.pvalues.to_dict()
        }
    except Exception as e:
        print(f"    Error fitting Model 3: {e}")
        results['complex_model'] = {'error': str(e)}
    
    # Model comparison
    print("  Comparing models...")
    model_comparison = {}
    
    for name, result in results.items():
        if 'error' not in result:
            model_comparison[name] = {
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Log-Likelihood': result['log_likelihood']
            }
    
    results['model_comparison'] = model_comparison
    
    # Extract random effects
    print("  Extracting random effects...")
    random_effects = {}
    
    for model_name, fitted_model in models.items():
        try:
            re = fitted_model.random_effects
            if re is not None:
                # Convert to serializable format
                re_dict = {}
                for group, effects in re.items():
                    if hasattr(effects, 'iloc'):
                        re_dict[str(group)] = effects.iloc[0] if len(effects) > 0 else 0
                    else:
                        re_dict[str(group)] = float(effects) if not pd.isna(effects) else 0
                random_effects[model_name] = re_dict
        except Exception as e:
            print(f"    Error extracting random effects for {model_name}: {e}")
            random_effects[model_name] = {}
    
    results['random_effects'] = random_effects
    
    return models, results, df

def analyze_group_effects(df, models, results):
    """Analyze specific group effects and performance patterns."""
    analysis = {}
    
    print("Analyzing group effects...")
    
    # Club performance analysis
    club_stats = df.groupby('club_analysis').agg({
        'finish_time': ['count', 'mean', 'std', 'median'],
        'age': 'mean',
        'gender': lambda x: (x == 'MALE').mean()
    }).round(3)
    
    club_stats.columns = ['count', 'mean_time', 'std_time', 'median_time', 'mean_age', 'pct_male']
    club_stats = club_stats[club_stats['count'] >= 5]  # Only clubs with 5+ members
    club_stats['cv_time'] = club_stats['std_time'] / club_stats['mean_time']  # Coefficient of variation
    
    analysis['club_performance'] = club_stats.to_dict()
    
    # Age group analysis
    age_group_stats = df.groupby('age_group').agg({
        'finish_time': ['count', 'mean', 'std', 'median'],
        'gender': lambda x: (x == 'MALE').mean()
    }).round(3)
    
    age_group_stats.columns = ['count', 'mean_time', 'std_time', 'median_time', 'pct_male']
    analysis['age_group_performance'] = age_group_stats.to_dict()
    
    # Gender performance by age
    gender_age_stats = df.groupby(['gender', 'age_group'])['finish_time'].agg(['count', 'mean', 'std']).round(3)
    analysis['gender_age_performance'] = gender_age_stats.to_dict()
    
    # Calculate intraclass correlation coefficients (ICC)
    if 'club_random' in results and 'error' not in results['club_random']:
        try:
            random_var = results['club_random']['random_effects_var']
            residual_var = results['club_random']['residual_var']
            
            if random_var is not None and residual_var is not None:
                icc_club = random_var / (random_var + residual_var)
                analysis['icc_club'] = icc_club
            else:
                analysis['icc_club'] = None
        except:
            analysis['icc_club'] = None
    
    # Performance predictions by group
    if 'club_random' in models:
        try:
            # Predict performance for different scenarios
            scenarios = [
                {'age': 30, 'gender': 'MALE'},
                {'age': 30, 'gender': 'FEMALE'},
                {'age': 45, 'gender': 'MALE'},
                {'age': 45, 'gender': 'FEMALE'},
            ]
            
            predictions = {}
            for scenario in scenarios:
                # Create a dummy dataframe for prediction
                pred_df = pd.DataFrame([scenario])
                pred_df['club_analysis'] = 'Average Club'
                
                try:
                    pred = models['club_random'].predict(pred_df)
                    predictions[f"Age{scenario['age']}_{scenario['gender']}"] = {
                        'log_time': float(pred.iloc[0]),
                        'time_seconds': float(np.exp(pred.iloc[0])),
                        'time_minutes': float(np.exp(pred.iloc[0]) / 60)
                    }
                except:
                    predictions[f"Age{scenario['age']}_{scenario['gender']}"] = None
            
            analysis['performance_predictions'] = predictions
        except Exception as e:
            print(f"Error in performance predictions: {e}")
            analysis['performance_predictions'] = {}
    
    return analysis

def save_results(models, results, analysis, df):
    """Save all analysis results."""
    
    # Save processed dataframe
    df.to_csv('data/processed/mixed_effects_data.csv', index=False)
    
    # Combine results and analysis
    full_results = {
        'model_results': results,
        'group_analysis': analysis,
        'data_summary': {
            'total_participants': len(df),
            'clubs_analyzed': df['club_analysis'].nunique(),
            'age_range': [float(df['age'].min()), float(df['age'].max())],
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'mean_finish_time_minutes': float(df['finish_time'].mean() / 60),
            'median_finish_time_minutes': float(df['finish_time'].median() / 60)
        }
    }
    
    # Save results to JSON (excluding model objects)
    with open('data/processed/mixed_effects_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save model objects separately
    with open('data/processed/mixed_effects_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Results saved to data/processed/")
    
    # Print summary
    print(f"\nMixed Effects Analysis Summary:")
    print(f"Total participants: {len(df)}")
    print(f"Clubs with 5+ members: {len([k for k, v in analysis['club_performance']['count'].items() if v >= 5])}")
    print(f"Age groups analyzed: {df['age_group'].nunique()}")
    
    if 'icc_club' in analysis and analysis['icc_club'] is not None:
        print(f"Intraclass correlation (club): {analysis['icc_club']:.3f}")
    
    # Model comparison
    if 'model_comparison' in results:
        print("\nModel Comparison (AIC - lower is better):")
        for model, metrics in results['model_comparison'].items():
            print(f"  {model}: AIC = {metrics['AIC']:.2f}")

def main():
    """Main execution function."""
    print("Starting Mixed Effects Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Fit models
    models, results, df_processed = fit_mixed_effects_models(df)
    
    # Analyze group effects
    analysis = analyze_group_effects(df_processed, models, results)
    
    # Save results
    save_results(models, results, analysis, df_processed)
    
    print("Mixed effects analysis complete!")

if __name__ == "__main__":
    main()