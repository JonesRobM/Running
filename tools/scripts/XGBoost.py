#!/usr/bin/env python3
"""
XGBoost Analysis for Race Performance Prediction
Advanced gradient boosting for high-accuracy performance prediction and feature analysis.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/processed', 'figures/xgboost']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_data():
    """Load race data and prepare comprehensive features for XGBoost."""
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
    
    # Age-related features
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    
    # Age groups with more granularity for XGBoost
    df['age_group_5yr'] = (df['age'] // 5) * 5  # 5-year age groups
    df['age_decade'] = (df['age'] // 10) * 10   # Decade groups
    
    # Gender encoding
    df['gender'] = df['gender'].str.upper()
    df['is_male'] = (df['gender'] == 'MALE').astype(int)
    df['is_female'] = (df['gender'] == 'FEMALE').astype(int)
    
    # Club features
    df['has_club'] = (~df['club'].isna() & (df['club'] != 'None') & (df['club'] != '')).astype(int)
    
    # Club size and performance features
    club_stats = df.groupby('club').agg({
        'finish_time': ['count', 'mean', 'std'],
        'age': 'mean'
    }).round(2)
    
    club_stats.columns = ['club_size', 'club_avg_time', 'club_time_std', 'club_avg_age']
    club_stats = club_stats.reset_index()
    
    # Merge club statistics back
    df = df.merge(club_stats, on='club', how='left')
    df['club_size'] = df['club_size'].fillna(0)
    df['club_avg_time'] = df['club_avg_time'].fillna(df['finish_time'].median())
    df['club_time_std'] = df['club_time_std'].fillna(df['finish_time'].std())
    df['club_avg_age'] = df['club_avg_age'].fillna(df['age'].median())
    
    # Large club indicator
    df['large_club'] = (df['club_size'] >= 10).astype(int)
    df['medium_club'] = ((df['club_size'] >= 5) & (df['club_size'] < 10)).astype(int)
    
    # Relative performance features
    df['age_vs_club_avg'] = df['age'] - df['club_avg_age']
    
    # Calculate 10km-based features if available
    if '10km_seconds' in df.columns:
        df['has_10km_split'] = (~df['10km_seconds'].isna()).astype(int)
        
        # Pacing features
        df['pace_10km'] = df['10km_seconds'] / 10000  # seconds per meter
        df['estimated_second_half'] = df['finish_time'] - df['10km_seconds']
        df['pace_second_half'] = df['estimated_second_half'] / 11097  # half marathon = 21.097km
        df['pace_overall'] = df['finish_time'] / 21097
        
        # Pacing ratios and strategies
        df['pace_ratio'] = df['pace_second_half'] / df['pace_10km']
        df['pace_differential'] = df['pace_second_half'] - df['pace_10km']
        df['negative_split'] = (df['pace_10km'] > df['pace_second_half']).astype(int)
        
        # Expected finish time based on 10km split
        df['expected_time_from_10k'] = df['10km_seconds'] * 2.1097  # Simple doubling
        df['time_prediction_error'] = df['finish_time'] - df['expected_time_from_10k']
        
        # Pacing efficiency metrics
        df['pacing_efficiency'] = 1 / df['pace_ratio']  # Higher is better
        df['energy_conservation'] = np.where(df['pace_ratio'] < 1, 1, 1/df['pace_ratio'])
    else:
        # Create dummy variables if no split data
        pace_features = ['has_10km_split', 'pace_10km', 'pace_ratio', 'negative_split', 
                        'time_prediction_error', 'pacing_efficiency']
        for feature in pace_features:
            df[feature] = 0
    
    # Position-based features (if available)
    if 'position' in df.columns:
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
        df['top_10_percent'] = (df['position_numeric'] <= len(df) * 0.1).astype(int)
        df['top_25_percent'] = (df['position_numeric'] <= len(df) * 0.25).astype(int)
    
    # Performance percentiles
    df['performance_rank'] = df['finish_time'].rank()
    df['performance_percentile'] = df['performance_rank'] / len(df) * 100
    
    # Interaction features
    df['age_gender_interaction'] = df['age'] * df['is_male']
    df['age_club_interaction'] = df['age'] * df['has_club']
    df['gender_club_interaction'] = df['is_male'] * df['has_club']
    
    if 'pace_10km' in df.columns:
        df['age_pace_interaction'] = df['age'] * df['pace_10km']
        df['gender_pace_interaction'] = df['is_male'] * df['pace_10km']
    
    # Binned age features for categorical treatment
    df['age_bin_young'] = (df['age'] < 30).astype(int)
    df['age_bin_middle'] = ((df['age'] >= 30) & (df['age'] < 50)).astype(int)
    df['age_bin_veteran'] = ((df['age'] >= 50) & (df['age'] < 65)).astype(int)
    df['age_bin_senior'] = (df['age'] >= 65).astype(int)
    
    return df

def prepare_features_advanced(df):
    """Prepare comprehensive feature matrix optimized for XGBoost."""
    
    # Define feature categories
    base_features = [
        'age', 'age_squared', 'age_cubed', 'is_male', 'is_female',
        'has_club', 'club_size', 'large_club', 'medium_club',
        'age_group_5yr', 'age_decade'
    ]
    
    club_features = [
        'club_avg_time', 'club_time_std', 'club_avg_age', 'age_vs_club_avg'
    ]
    
    pacing_features = [
        'has_10km_split', '10km_seconds', 'pace_10km', 'pace_ratio',
        'pace_differential', 'negative_split', 'time_prediction_error',
        'pacing_efficiency'
    ]
    
    interaction_features = [
        'age_gender_interaction', 'age_club_interaction', 'gender_club_interaction'
    ]
    
    if 'pace_10km' in df.columns and df['pace_10km'].notna().sum() > 0:
        interaction_features.extend(['age_pace_interaction', 'gender_pace_interaction'])
    
    binned_features = [
        'age_bin_young', 'age_bin_middle', 'age_bin_veteran', 'age_bin_senior'
    ]
    
    # Combine all features
    all_features = base_features + club_features + pacing_features + interaction_features + binned_features
    
    # Add top club indicators (for clubs with >20 members)
    top_clubs = df['club'].value_counts().head(15).index
    for club in top_clubs:
        club_feature = f'club_{club.replace(" ", "_").replace(".", "").replace("-", "_")}'
        df[club_feature] = (df['club'] == club).astype(int)
        all_features.append(club_feature)
    
    # Create feature matrix
    feature_df = df[all_features].copy()
    
    # Target variable
    target = df['finish_time']
    
    # Remove rows with missing target or critical features
    complete_cases = target.notna() & df['age'].notna()
    
    feature_df_clean = feature_df[complete_cases]
    target_clean = target[complete_cases]
    
    # Fill remaining missing values with appropriate defaults
    for col in feature_df_clean.columns:
        if feature_df_clean[col].dtype in ['float64', 'int64']:
            feature_df_clean[col] = feature_df_clean[col].fillna(0)
        else:
            feature_df_clean[col] = feature_df_clean[col].fillna('Unknown')
    
    return feature_df_clean, target_clean, complete_cases

def train_xgboost_models(X, y):
    """Train and optimize XGBoost models with comprehensive hyperparameter tuning."""
    
    # Split data with stratification based on performance quartiles
    y_quartiles = pd.qcut(y, q=4, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_quartiles
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    models = {}
    results = {}
    
    # 1. Baseline XGBoost model
    print("Training baseline XGBoost...")
    
    xgb_baseline = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_baseline.fit(X_train, y_train)
    models['baseline'] = xgb_baseline
    
    # Cross-validation
    cv_scores = cross_val_score(xgb_baseline, X_train, y_train, cv=5, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    
    # Predictions
    y_pred_train_base = xgb_baseline.predict(X_train)
    y_pred_test_base = xgb_baseline.predict(X_test)
    
    results['baseline'] = {
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std(),
        'train_mae': mean_absolute_error(y_train, y_pred_train_base),
        'test_mae': mean_absolute_error(y_test, y_pred_test_base),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_base)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_base)),
        'train_r2': r2_score(y_train, y_pred_train_base),
        'test_r2': r2_score(y_test, y_pred_test_base),
        'feature_importance': dict(zip(X.columns, xgb_baseline.feature_importances_))
    }
    
    # 2. Hyperparameter optimization
    print("Optimizing hyperparameters...")
    
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    # Use smaller parameter grid for faster execution
    xgb_tuned = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        xgb_tuned, param_grid, cv=3,
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
    
    # 3. Early stopping model for overfitting control
    print("Training with early stopping...")
    
    xgb_early = xgb.XGBRegressor(
        **grid_search.best_params_,
        n_estimators=1000,  # Large number, will stop early
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    # Use part of training set for validation
    X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    xgb_early.fit(
        X_train_es, y_train_es,
        eval_set=[(X_val_es, y_val_es)],
        verbose=False
    )
    
    models['early_stopping'] = xgb_early
    
    # Evaluate early stopping model
    y_pred_train_es = xgb_early.predict(X_train)
    y_pred_test_es = xgb_early.predict(X_test)
    
    results['early_stopping'] = {
        'best_iteration': xgb_early.best_iteration,
        'train_mae': mean_absolute_error(y_train, y_pred_train_es),
        'test_mae': mean_absolute_error(y_test, y_pred_test_es),
        'train_r2': r2_score(y_train, y_pred_train_es),
        'test_r2': r2_score(y_test, y_pred_test_es),
        'feature_importance': dict(zip(X.columns, xgb_early.feature_importances_))
    }
    
    # 4. Permutation importance analysis
    print("Computing permutation importance...")
    
    best_model = grid_search.best_estimator_
    perm_importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    permutation_results = {
        'importances_mean': dict(zip(X.columns, perm_importance.importances_mean)),
        'importances_std': dict(zip(X.columns, perm_importance.importances_std))
    }
    
    results['permutation_importance'] = permutation_results
    
    # Store predictions for analysis
    results['predictions'] = {
        'y_test': y_test.tolist(),
        'y_pred_baseline': y_pred_test_base.tolist(),
        'y_pred_optimized': y_pred_test_opt.tolist(),
        'y_pred_early_stopping': y_pred_test_es.tolist()
    }
    
    # Feature importance comparison
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': grid_search.best_estimator_.feature_importances_,
        'permutation_importance': perm_importance.importances_mean
    }).sort_values('xgb_importance', ascending=False)
    
    results['feature_importance_comparison'] = feature_importance_df.to_dict()
    
    return models, results, (X_train, X_test, y_train, y_test)

def analyze_model_insights(models, results, X, y):
    """Extract insights and interpretations from XGBoost models."""
    analysis = {}
    
    print("Analyzing model insights...")
    
    best_model = models['optimized']
    
    # 1. Feature importance analysis
    feature_imp = pd.DataFrame(results['feature_importance_comparison'])
    top_features = feature_imp.head(15)
    analysis['top_features'] = top_features.to_dict()
    
    # 2. Performance by feature categories
    feature_categories = {
        'Age Features': [f for f in X.columns if 'age' in f.lower()],
        'Gender Features': [f for f in X.columns if 'gender' in f.lower() or 'male' in f.lower() or 'female' in f.lower()],
        'Club Features': [f for f in X.columns if 'club' in f.lower()],
        'Pacing Features': [f for f in X.columns if 'pace' in f.lower() or '10km' in f.lower()],
        'Interaction Features': [f for f in X.columns if 'interaction' in f.lower()]
    }
    
    category_importance = {}
    for category, features in feature_categories.items():
        category_features = [f for f in features if f in feature_imp['feature'].values]
        if category_features:
            category_df = feature_imp[feature_imp['feature'].isin(category_features)]
            category_importance[category] = {
                'total_importance': category_df['xgb_importance'].sum(),
                'mean_importance': category_df['xgb_importance'].mean(),
                'top_feature': category_df.iloc[0]['feature'] if len(category_df) > 0 else None,
                'feature_count': len(category_features)
            }
    
    analysis['category_importance'] = category_importance
    
    # 3. Prediction scenarios
    scenarios = []
    feature_means = X.mean()
    
    # Scenario 1: Young fast runner
    scenario1 = feature_means.copy()
    scenario1['age'] = 25
    scenario1['age_squared'] = 625
    scenario1['age_cubed'] = 15625
    scenario1['is_male'] = 1
    scenario1['is_female'] = 0
    scenario1['has_club'] = 1
    scenario1['large_club'] = 1
    if '10km_seconds' in X.columns:
        scenario1['10km_seconds'] = 2400  # 40 minutes for 10k
        scenario1['has_10km_split'] = 1
    scenarios.append(('Young Elite Male', scenario1))
    
    # Scenario 2: Middle-aged recreational runner
    scenario2 = feature_means.copy()
    scenario2['age'] = 45
    scenario2['age_squared'] = 2025
    scenario2['age_cubed'] = 91125
    scenario2['is_male'] = 0
    scenario2['is_female'] = 1
    scenario2['has_club'] = 0
    scenarios.append(('Middle-aged Recreational Female', scenario2))
    
    # Scenario 3: Veteran with club
    scenario3 = feature_means.copy()
    scenario3['age'] = 55
    scenario3['age_squared'] = 3025
    scenario3['age_cubed'] = 166375
    scenario3['is_male'] = 1
    scenario3['is_female'] = 0
    scenario3['has_club'] = 1
    scenario3['large_club'] = 1
    if '10km_seconds' in X.columns:
        scenario3['10km_seconds'] = 3000  # 50 minutes for 10k
        scenario3['has_10km_split'] = 1
    scenarios.append(('Veteran Club Male', scenario3))
    
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
    
    # 4. Model performance insights
    best_results = results['optimized']
    analysis['model_performance'] = {
        'test_mae_minutes': best_results['test_mae'] / 60,
        'test_r2': best_results['test_r2'],
        'overfitting_assessment': {
            'r2_gap': best_results['train_r2'] - best_results['test_r2'],
            'mae_gap': best_results['test_mae'] - best_results['train_mae'],
            'interpretation': 'Low overfitting' if (best_results['train_r2'] - best_results['test_r2']) < 0.05 else 'Moderate overfitting'
        },
        'early_stopping_benefit': {
            'iterations_used': results['early_stopping']['best_iteration'],
            'performance_difference': abs(results['early_stopping']['test_mae'] - best_results['test_mae'])
        }
    }
    
    # 5. Feature interaction insights
    if 'age_gender_interaction' in X.columns:
        interaction_importance = feature_imp[feature_imp['feature'].str.contains('interaction')]['xgb_importance'].sum()
        analysis['interaction_effects'] = {
            'total_interaction_importance': interaction_importance,
            'percentage_of_total': interaction_importance / feature_imp['xgb_importance'].sum() * 100
        }
    
    return analysis

def save_results(models, results, analysis, X, y):
    """Save all XGBoost analysis results."""
    
    # Save feature matrix and target
    feature_target_df = X.copy()
    feature_target_df['finish_time'] = y
    feature_target_df.to_csv('data/processed/xgboost_data.csv', index=False)
    
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
    with open('data/processed/xgboost_results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    
    # Save model objects
    with open('data/processed/xgboost_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("Results saved to data/processed/")
    
    # Print summary
    print(f"\nXGBoost Analysis Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Features used: {len(X.columns)}")
    print(f"Best model R²: {results['optimized']['test_r2']:.3f}")
    print(f"Best model MAE: {results['optimized']['test_mae']/60:.1f} minutes")
    print(f"Best parameters: {results['optimized']['best_params']}")
    
    print(f"\nTop 5 Most Important Features:")
    feature_imp = pd.DataFrame(results['feature_importance_comparison'])
    for _, row in feature_imp.head().iterrows():
        print(f"  {row['feature']}: {row['xgb_importance']:.3f}")

def main():
    """Main execution function."""
    print("Starting XGBoost Analysis...")
    
    # Setup directories
    setup_directories()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Prepare features
    print("Preparing advanced features...")
    X, y, complete_cases = prepare_features_advanced(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    # Train models
    models, results, splits = train_xgboost_models(X, y)
    
    # Analyze results
    print("Analyzing model insights...")
    analysis = analyze_model_insights(models, results, X, y)
    
    # Save results
    save_results(models, results, analysis, X, y)
    
    print("XGBoost analysis complete!")

if __name__ == "__main__":
    main()