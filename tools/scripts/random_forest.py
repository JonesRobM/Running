#!/usr/bin/env python3
"""
Random Forest Analysis for Race Performance Prediction
Predicts finish times and identifies key performance factors using Random Forest.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/random_forest']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare features for Random Forest."""
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
    
    # Target variable: finish time (chip time preferred)
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
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                           bins=[0, 30, 40, 50, 60, 100], 
                           labels=['Under30', '30-39', '40-49', '50-59', '60Plus'])
    
    # Gender encoding
    df['gender'] = df['gender'].str.upper()
    df['is_male'] = (df['gender'] == 'MALE').astype(int)
    
    # Club features
    df['has_club'] = (~df['club'].isna() & (df['club'] != 'None') & (df['club'] != '')).astype(int)
    
    # Calculate 10km pace if available
    if '10km_seconds' in df.columns:
        df['pace_10km'] = df['10km_seconds'] / 10000  # seconds per meter
        df['has_10km_split'] = (~df['10km_seconds'].isna()).astype(int)
        
        # Estimate final pace and pace degradation
        df['estimated_final_pace'] = df['finish_time'] / 21097  # half marathon distance
        df['pace_degradation'] = (df['estimated_final_pace'] - df['pace_10km']) / df['pace_10km']
    else:
        df['has_10km_split'] = 0
    
    # Position-based features
    if 'position' in df.columns:
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
    
    # Create performance categories for analysis
    df['performance_quartile'] = pd.qcut(df['finish_time'], 
                                       q=4, 
                                       labels=['Fast', 'Medium-Fast', 'Medium-Slow', 'Slow'])
    
    # Club size effect
    club_sizes = df['club'].value_counts()
    df['club_size'] = df['club'].map(club_sizes).fillna(0)
    df['large_club'] = (df['club_size'] >= 10).astype(int)
    
    # Age-gender interaction
    df['age_male_interaction'] = df['age'] * df['is_male']
    
    return df

def prepare_features(df):
    """Prepare feature matrix and target variable."""
    
    # Define base features
    base_features = [
        'age', 'is_male', 'has_club', 'has_10km_split', 
        'club_size', 'large_club', 'age_male_interaction'
    ]
    
    # Add 10km split features if available
    if '10km_seconds' in df.columns:
        base_features.extend(['10km_seconds', 'pace_10km'])
    
    # Add age group dummies
    age_group_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
    
    # Combine features
    feature_df = df[base_features].copy()
    feature_df = pd.concat([feature_df, age_group_dummies], axis=1)
    
    # Handle categorical club encoding (top clubs only)
    top_clubs = df['club'].value_counts().head(20).index
    for club in top_clubs:
        feature_df[f'club_{club.replace(" ", "_")}'] = (df['club'] == club).astype(int)
    
    # Target variable
    target = df['finish_time']
    
    # Remove rows with missing target or key features
    complete_cases = target.notna() & df['age'].notna()
    
    feature_df_clean = feature_df[complete_cases]
    target_clean = target[complete_cases]
    
    # Fill remaining missing values
    feature_df_clean = feature_df_clean.fillna(0)
    
    return feature_df_clean, target_clean, complete_cases

def train_random_forest_models(X, y):
    """Train and evaluate Random Forest models."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    models = {}
    results = {}
    
    # Basic Random Forest
    print("Training basic Random Forest...")
    rf_basic = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_basic.fit(X_train, y_train)
    models['basic'] = rf_basic
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_basic, X_train, y_train, cv=5, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # Predictions
    y_pred_train = rf_basic.predict(X_train)
    y_pred_test = rf_basic.predict(X_test)
    
    results['basic'] = {
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std(),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'feature_importance': dict(zip(X.columns, rf_basic.feature_importances_))
    }
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    # Use smaller grid for faster execution
    rf_grid = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf_grid, param_grid, cv=3, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    models['optimized'] = grid_search.best_estimator_
    
    # Evaluate optimized model
    y_pred_train_opt = grid_search.best_estimator_.predict(X_train)
    y_pred_test_opt = grid_search.best_estimator_.predict(X_test)
    
    cv_scores_opt = cross_val_score(grid_search.best_estimator_, X_train, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    results['optimized'] = {
        'best_params': grid_search.best_params_,
        'cv_mae_mean': -cv_scores_opt.mean(),
        'cv_mae_std': cv_scores_opt.std(),
        'train_mae': mean_absolute_error(y_train, y_pred_train_opt),
        'test_mae': mean_absolute_error(y_test, y_pred_test_opt),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_opt)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_opt)),
        'train_r2': r2_score(y_train, y_pred_train_opt),
        'test_r2': r2_score(y_test, y_pred_test_opt),
        'feature_importance': dict(zip(X.columns, grid_search.best_estimator_.feature_importances_))
    }
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance_basic': rf_basic.feature_importances_,
        'importance_optimized': grid_search.best_estimator_.feature_importances_
    }).sort_values('importance_optimized', ascending=False)
    
    results['feature_importance_df'] = feature_importance.to_dict()
    
    # Store test predictions for analysis
    results['predictions'] = {
        'y_test': y_test.tolist(),
        'y_pred_basic': y_pred_test.tolist(),
        'y_pred_optimized': y_pred_test_opt.tolist()
    }
    
    return models, results, (X_train, X_test, y_train, y_test)

def analyze_performance_factors(models, results, X, y):
    """Analyze key performance factors and model insights."""
    analysis = {}
    
    # Feature importance ranking
    feature_imp = pd.DataFrame(results['feature_importance_df'])
    top_features = feature_imp.head(10)
    analysis['top_10_features'] = top_features.to_dict()
    
    # Performance prediction examples
    best_model = models['optimized']
    
    # Create prediction scenarios
    scenarios = []
    
    # Get feature means for baseline
    feature_means = X.mean()
    
    # Scenario 1: Young male with club
    scenario1 = feature_means.copy()
    scenario1['age'] = 25
    scenario1['is_male'] = 1
    scenario1['has_club'] = 1
    scenario1['age_group_Under30'] = 1
    scenario1[[col for col in X.columns if col.startswith('age_group_') and col != 'age_group_Under30']] = 0
    scenarios.append(('Young Male with Club', scenario1))
    
    # Scenario 2: Middle-aged female without club
    scenario2 = feature_means.copy()
    scenario2['age'] = 45
    scenario2['is_male'] = 0
    scenario2['has_club'] = 0
    scenario2['age_group_40-49'] = 1
    scenario2[[col for col in X.columns if col.startswith('age_group_') and col != 'age_group_40-49']] = 0
    scenarios.append(('Middle-aged Female without Club', scenario2))
    
    # Scenario 3: Older male with fast 10k split
    scenario3 = feature_means.copy()
    scenario3['age'] = 55
    scenario3['is_male'] = 1
    scenario3['has_club'] = 1
    if '10km_seconds' in X.columns:
        scenario3['10km_seconds'] = 2400  # 40 minutes for 10k
        scenario3['pace_10km'] = 0.24  # seconds per meter
    scenario3['age_group_50-59'] = 1
    scenario3[[col for col in X.columns if col.startswith('age_group_') and col != 'age_group_50-59']] = 0
    scenarios.append(('Older Male with Fast 10k', scenario3))
    
    # Make predictions
    predictions = {}
    for name, scenario in scenarios:
        try:
            pred = best_model.predict(scenario.values.reshape(1, -1))[0]
            predictions[name] = {
                'predicted_seconds': pred,
                'predicted_minutes': pred / 60,
                'predicted_time_formatted': f"{int(pred//3600)}:{int((pred%3600)//60):02d}:{int(pred%60):02d}"
            }
        except Exception as e:
            predictions[name] = {'error': str(e)}
    
    analysis['scenario_predictions'] = predictions
    
    # Performance factor insights
    insights = {}
    
    # Age effect
    if 'age' in results['optimized']['feature_importance']:
        age_importance = results['optimized']['feature_importance']['age']
        insights['age_effect'] = {
            'importance': age_importance,
            'interpretation': 'Higher importance indicates age is a strong predictor of performance'
        }
    
    # Gender effect
    if 'is_male' in results['optimized']['feature_importance']:
        gender_importance = results['optimized']['feature_importance']['is_male']
        insights['gender_effect'] = {
            'importance': gender_importance,
            'interpretation': 'Indicates gender differences in performance'
        }
    
    # Club effect
    if 'has_club' in results['optimized']['feature_importance']:
        club_importance = results['optimized']['feature_importance']['has_club']
        insights['club_effect'] = {
            'importance': club_importance,
            'interpretation': 'Shows impact of club membership on performance'
        }
    
    # 10k split effect
    if '10km_seconds' in results['optimized']['feature_importance']:
        split_importance = results['optimized']['feature_importance']['10km_seconds']
        insights['split_time_effect'] = {
            'importance': split_importance,
            'interpretation': '10k split time as predictor of final performance'
        }
    
    analysis['performance_insights'] = insights
    
    # Model performance summary
    best_results = results['optimized']
    analysis['model_performance'] = {
        'test_mae_minutes': best_results['test_mae'] / 60,
        'test_r2': best_results['test_r2'],
        'interpretation': {
            'mae': f"On average, predictions are within {best_results['test_mae']/60:.1f} minutes of actual time",
            'r2': f"Model explains {best_results['test_r2']*100:.1f}% of variance in finish times"
        }
    }
    
    return analysis

def save_results(models, results, analysis, X, y):
    """Save all analysis results."""
    
    # Save feature matrix and target
    feature_target_df = X.copy()
    feature_target_df['finish_time'] = y
    feature_target_df.to_csv('data/processed/random_forest_data.csv', index=False)
    
    # Combine all results
    full_results = {
        'model_results': results,
        'analysis': analysis,
        'data_summary': {
            'total_samples': len(X),
            'num_features': len(X.columns),
            'target_mean_minutes': float(y.mean() / 60),
            'target_std_minutes': float(y.std() / 60),
            'feature_list': X.columns.tolist()
        }
    }
    
    # Save results to JSON
    with open('data/processed/random_forest_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save model objects
    with open('data/processed/random_forest_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Results saved to data/processed/")
    
    # Print summary
    print(f"\nRandom Forest Analysis Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Features used: {len(X.columns)}")
    print(f"Best model R²: {results['optimized']['test_r2']:.3f}")
    print(f"Best model MAE: {results['optimized']['test_mae']/60:.1f} minutes")
    
    print(f"\nTop 5 Most Important Features:")
    feature_imp = pd.DataFrame(results['feature_importance_df']).head()
    for _, row in feature_imp.iterrows():
        print(f"  {row['feature']}: {row['importance_optimized']:.3f}")

def main():
    """Main execution function."""
    print("Starting Random Forest Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Prepare features
    print("Preparing features...")
    X, y, complete_cases = prepare_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Train models
    models, results, splits = train_random_forest_models(X, y)
    
    # Analyze results
    print("Analyzing performance factors...")
    analysis = analyze_performance_factors(models, results, X, y)
    
    # Save results
    save_results(models, results, analysis, X, y)
    
    print("Random Forest analysis complete!")

if __name__ == "__main__":
    main()