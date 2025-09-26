"""
Integration layer for Greta Core Engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

from greta_core import ingestion, preprocessing, hypothesis_search, statistical_analysis, narrative_generation
from ..config import GretaConfig


def run_analysis_pipeline(config: GretaConfig, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline using Greta Core.

    Args:
        config: GretaConfig instance with analysis parameters.
        overrides: Optional parameter overrides.

    Returns:
        Dictionary containing analysis results.
    """
    if overrides:
        # Apply overrides to config (simplified)
        pass

    # Step 1: Data Ingestion
    with tqdm(total=1, desc="Data Ingestion") as pbar:
        if config.data.type == 'csv':
            df = ingestion.load_csv(config.data.source)
        elif config.data.type == 'excel':
            df = ingestion.load_excel(config.data.source, sheet_name=config.data.sheet_name)
        else:
            raise ValueError(f"Unsupported data type: {config.data.type}")

        # Validate data
        warnings = ingestion.validate_data(df)
        if warnings:
            print("Data validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        # Determine target column
        if config.data.target_column is None:
            # Assume last column is target
            target_col = df.columns[-1]
            print(f"No target column specified, using: {target_col}")
        else:
            target_col = config.data.target_column
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")

        # Separate features and target
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # Encode categorical target if necessary
        if not np.issubdtype(y.dtype, np.number):
            unique_vals = np.unique(y)
            if len(unique_vals) == 2:
                # Binary encoding
                y = np.where(y == unique_vals[0], 0, 1).astype(float)
            else:
                # Multi-class, use label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y).astype(float)

        # Select only numeric columns for features
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        X_df = X_df[numeric_cols]
        feature_names = list(X_df.columns)
        pbar.update(1)

    # Step 2: Preprocessing
    with tqdm(total=4, desc="Preprocessing") as main_pbar:
        # Sub-stage 1: Data Profiling
        with tqdm(total=1, desc="Data Profiling", leave=False) as sub_pbar:
            profile = preprocessing.profile_data(X_df, progress_callback=lambda: sub_pbar.update(1))
            print(f"Data shape: {profile['shape']}")
        main_pbar.update(1)

        # Sub-stage 2: Missing Value Handling
        with tqdm(total=1, desc="Missing Value Handling", leave=False) as sub_pbar:
            X_clean = preprocessing.handle_missing_values(
                X_df,
                strategy=config.preprocessing.missing_strategy,
                progress_callback=lambda: sub_pbar.update(1)
            )
        main_pbar.update(1)

        # Sub-stage 3: Outlier Detection
        with tqdm(total=1, desc="Outlier Detection", leave=False) as sub_pbar:
            outliers = preprocessing.detect_outliers(
                X_clean,
                method=config.preprocessing.outlier_method,
                threshold=config.preprocessing.outlier_threshold,
                progress_callback=lambda: sub_pbar.update(1)
            )

            # Remove outliers if any detected
            if any(outliers.values()):
                X_clean = preprocessing.remove_outliers(X_clean, outliers)
                y = y[~np.isin(np.arange(len(y)), list(set().union(*outliers.values())))]
                print(f"Removed {sum(len(indices) for indices in outliers.values())} outlier rows")
        main_pbar.update(1)

        # Sub-stage 4: Feature Encoding
        with tqdm(total=1, desc="Feature Encoding", leave=False) as sub_pbar:
            # Normalize data types
            if config.preprocessing.normalize_types:
                X_clean = preprocessing.normalize_data_types(X_clean, progress_callback=lambda: sub_pbar.update(0.5) if sub_pbar.n < 1 else None)

            # Feature engineering
            if config.preprocessing.feature_engineering:
                X_clean = preprocessing.basic_feature_engineering(X_clean, progress_callback=lambda: sub_pbar.update(0.5) if sub_pbar.n < 1 else None)

            # Ensure progress bar completes
            if sub_pbar.n < 1:
                sub_pbar.update(1 - sub_pbar.n)

        # Convert to numpy arrays
        X = X_clean.values
        feature_names = list(X_clean.columns)

        # Identify and exclude identifier columns
        identifier_cols = preprocessing.identify_identifier_columns(X_clean)
        if identifier_cols:
            print(f"Excluding identifier columns: {identifier_cols}")
            cols_to_keep = [col for col in X_clean.columns if col not in identifier_cols]
            X_clean = X_clean[cols_to_keep]
            X = X_clean.values
            feature_names = list(X_clean.columns)

        print(f"Processed data shape: {X.shape}")
        main_pbar.update(1)

    # Step 3: Hypothesis Search
    with tqdm(total=config.hypothesis_search.num_generations, desc="Hypothesis Search Generations") as pbar:
        hypotheses = hypothesis_search.generate_hypotheses(
            X, y,
            pop_size=config.hypothesis_search.pop_size,
            num_generations=config.hypothesis_search.num_generations,
            cx_prob=config.hypothesis_search.cx_prob,
            mut_prob=config.hypothesis_search.mut_prob,
            progress_callback=lambda: pbar.update(1)
        )

    print(f"Generated {len(hypotheses)} hypotheses")

    # Step 4: Statistical Analysis
    print("Performing statistical analysis...")
    stat_results = []
    for hyp in hypotheses:
        selected_features = X[:, hyp['features']]
        result = statistical_analysis.perform_statistical_test(selected_features, y)
        stat_results.append(result)

    # Step 5: Narrative Generation
    print("Generating narratives...")
    summary_narrative = narrative_generation.generate_summary_narrative(hypotheses, feature_names, X, y, metadata={
        'data_shape': X.shape,
        'target_column': target_col,
        'feature_names': feature_names
    })
    detailed_report = narrative_generation.create_report(hypotheses, feature_names, stat_results)

    # Compile results
    results = {
        'metadata': {
            'data_shape': X.shape,
            'target_column': target_col,
            'feature_names': feature_names,
            'num_hypotheses': len(hypotheses),
            'config': config.dict()
        },
        'data_profile': profile,
        'hypotheses': hypotheses,
        'statistical_results': stat_results,
        'summary_narrative': summary_narrative,
        'detailed_report': detailed_report
    }

    return results