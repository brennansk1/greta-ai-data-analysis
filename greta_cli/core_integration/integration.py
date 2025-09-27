"""
Integration layer for Greta Core Engine.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import time
import logging

from greta_core import ingestion, preprocessing, hypothesis_search, statistical_analysis, narrative_generation, causal_analysis
from ..config import GretaConfig

logger = logging.getLogger(__name__)


def run_analysis_pipeline(config: GretaConfig, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the complete analysis pipeline using Greta Core.

    Args:
        config: GretaConfig instance with analysis parameters.
        overrides: Optional parameter overrides.

    Returns:
        Dictionary containing analysis results.
    """
    logger.info("Starting analysis pipeline execution")
    logger.info(f"Config: data_source={config.data.source}, target={config.data.target_column}")
    logger.info(f"Overrides: {overrides}")

    if overrides:
        logger.info("Applying parameter overrides to config")
        # Apply overrides to config dynamically
        for key, value in overrides.items():
            if key == 'hypothesis_search' and isinstance(value, dict):
                # Handle nested hypothesis_search overrides
                for sub_key, sub_value in value.items():
                    if hasattr(config.hypothesis_search, sub_key):
                        setattr(config.hypothesis_search, sub_key, sub_value)
                        logger.info(f"Applied hypothesis_search override: {sub_key} = {sub_value}")
                    else:
                        logger.warning(f"Unknown hypothesis_search override key: {sub_key}")
            elif hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Applied override: {key} = {value}")
            elif key in ['bootstrap_iterations', 'encoding_method', 'pre_filter_fraction', 'use_causal_prioritization', 'adaptive_params', 'diversity_threshold', 'convergence_threshold', 'n_processes']:
                # These are passed directly to generate_hypotheses
                logger.info(f"Override {key} will be passed to generate_hypotheses")
            else:
                logger.warning(f"Unknown override key: {key}")

    # Step 1: Data Ingestion
    logger.info("Starting data ingestion phase")
    start_time = time.time()
    with tqdm(total=1, desc="Data Ingestion") as pbar:
        logger.info(f"Loading data from {config.data.source} (type: {config.data.type})")
        if config.data.type == 'csv':
            df = ingestion.load_csv(config.data.source)
        elif config.data.type == 'excel':
            df = ingestion.load_excel(config.data.source, sheet_name=config.data.sheet_name)
        else:
            raise ValueError(f"Unsupported data type: {config.data.type}")

        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Validate data
        warnings = ingestion.validate_data(df)
        if warnings:
            logger.warning(f"Data validation warnings: {warnings}")
            print("Data validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        # Determine target column
        if config.data.target_column is None:
            # Assume last column is target
            target_col = df.columns[-1]
            logger.info(f"No target column specified, using: {target_col}")
            print(f"No target column specified, using: {target_col}")
        else:
            target_col = config.data.target_column
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")

        logger.info(f"Target column: {target_col}")

        # Separate features and target
        logger.info("Separating features and target")
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])

        # Encode categorical target if necessary
        if not np.issubdtype(y.dtype, np.number):
            logger.info("Encoding categorical target variable")
            unique_vals = np.unique(y)
            if len(unique_vals) == 2:
                # Binary encoding
                y = np.where(y == unique_vals[0], 0, 1).astype(float)
                logger.info(f"Binary encoding applied: {unique_vals[0]} -> 0, {unique_vals[1]} -> 1")
            else:
                # Multi-class, use label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y).astype(float)
                logger.info(f"Label encoding applied for {len(unique_vals)} classes")

        # Select only numeric columns for features
        logger.info("Selecting numeric feature columns")
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        X_df = X_df[numeric_cols]
        feature_names = list(X_df.columns)
        logger.info(f"Selected {len(feature_names)} numeric features: {feature_names}")
        pbar.update(1)
    ingestion_time = time.time() - start_time
    logger.info(f"Data ingestion completed in {ingestion_time:.2f} seconds")
    print(f"Data Ingestion completed in {ingestion_time:.2f} seconds")

    # Step 2: Preprocessing
    logger.info("Starting preprocessing phase")
    preprocessing_start = time.time()
    with tqdm(total=4, desc="Preprocessing") as main_pbar:
        # Sub-stage 1: Data Profiling
        logger.info("Starting data profiling")
        with tqdm(total=1, desc="Data Profiling", leave=False) as sub_pbar:
            profile = preprocessing.profile_data(X_df, progress_callback=lambda: sub_pbar.update(1))
            logger.info(f"Data profiling completed. Shape: {profile['shape']}")
            print(f"Data shape: {profile['shape']}")
        main_pbar.update(1)

        # Sub-stage 2: Missing Value Handling
        logger.info(f"Starting missing value handling with strategy: {config.preprocessing.missing_strategy}")
        with tqdm(total=1, desc="Missing Value Handling", leave=False) as sub_pbar:
            X_clean = preprocessing.handle_missing_values(
                X_df,
                strategy=config.preprocessing.missing_strategy,
                progress_callback=lambda: sub_pbar.update(1)
            )
            logger.info("Missing value handling completed")
        main_pbar.update(1)

        # Sub-stage 3: Outlier Detection
        logger.info(f"Starting outlier detection with method: {config.preprocessing.outlier_method}")
        with tqdm(total=1, desc="Outlier Detection", leave=False) as sub_pbar:
            outliers = preprocessing.detect_outliers(
                X_clean,
                method=config.preprocessing.outlier_method,
                threshold=config.preprocessing.outlier_threshold,
                progress_callback=lambda: sub_pbar.update(1)
            )

            # Remove outliers if any detected
            if any(outliers.values()):
                outlier_count = sum(len(indices) for indices in outliers.values())
                logger.info(f"Removing {outlier_count} outlier rows")
                X_clean = preprocessing.remove_outliers(X_clean, outliers)
                y = y[~np.isin(np.arange(len(y)), list(set().union(*outliers.values())))]
                print(f"Removed {outlier_count} outlier rows")
        main_pbar.update(1)

        # Sub-stage 4: Feature Encoding
        logger.info("Starting feature encoding and engineering")
        with tqdm(total=1, desc="Feature Encoding", leave=False) as sub_pbar:
            # Normalize data types
            if config.preprocessing.normalize_types:
                logger.info("Normalizing data types")
                X_clean = preprocessing.normalize_data_types(X_clean, progress_callback=lambda: sub_pbar.update(0.5) if sub_pbar.n < 1 else None)

            # Feature engineering
            if config.preprocessing.feature_engineering:
                logger.info("Performing basic feature engineering")
                X_clean = preprocessing.basic_feature_engineering(X_clean, progress_callback=lambda: sub_pbar.update(0.5) if sub_pbar.n < 1 else None)

            # Ensure progress bar completes
            if sub_pbar.n < 1:
                sub_pbar.update(1 - sub_pbar.n)

        # Convert to numpy arrays
        logger.info("Converting to numpy arrays")
        X = X_clean.values
        feature_names = list(X_clean.columns)

        # Identify and exclude identifier columns
        identifier_cols = preprocessing.identify_identifier_columns(X_clean)
        if identifier_cols:
            logger.info(f"Excluding identifier columns: {identifier_cols}")
            print(f"Excluding identifier columns: {identifier_cols}")
            cols_to_keep = [col for col in X_clean.columns if col not in identifier_cols]
            X_clean = X_clean[cols_to_keep]
            X = X_clean.values
            feature_names = list(X_clean.columns)

        logger.info(f"Preprocessing completed. Final data shape: {X.shape}, features: {len(feature_names)}")
        print(f"Processed data shape: {X.shape}")
        main_pbar.update(1)
    preprocessing_time = time.time() - preprocessing_start
    logger.info(f"Preprocessing phase completed in {preprocessing_time:.2f} seconds")
    print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")

    # Step 3: Hypothesis Search
    logger.info("Starting hypothesis search phase")
    logger.info(f"GA parameters: pop_size={config.hypothesis_search.pop_size}, generations={config.hypothesis_search.num_generations}, cx_prob={config.hypothesis_search.cx_prob}, mut_prob={config.hypothesis_search.mut_prob}")
    logger.info(f"Data dimensions for GA: X.shape={X.shape}, y.shape={y.shape}")

    # Prepare additional parameters from overrides
    ga_kwargs = {
        'pop_size': config.hypothesis_search.pop_size,
        'num_generations': config.hypothesis_search.num_generations,
        'cx_prob': config.hypothesis_search.cx_prob,
        'mut_prob': config.hypothesis_search.mut_prob,
        'progress_callback': lambda: None  # We'll handle progress manually
    }

    # Apply overrides for additional parameters
    if overrides:
        for key in ['bootstrap_iterations', 'encoding_method', 'pre_filter_fraction', 'use_causal_prioritization', 'adaptive_params', 'diversity_threshold', 'convergence_threshold', 'n_processes']:
            if key in overrides:
                ga_kwargs[key] = overrides[key]
                logger.info(f"Applied GA override: {key} = {overrides[key]}")

    logger.info(f"Final GA kwargs: {ga_kwargs}")

    hypothesis_start = time.time()
    with tqdm(total=config.hypothesis_search.num_generations, desc="Hypothesis Search Generations") as pbar:
        logger.info("Calling hypothesis_search.generate_hypotheses")
        start_gen_time = time.time()

        # Override the progress callback to update our progress bar
        ga_kwargs['progress_callback'] = lambda: pbar.update(1)

        # Pass feature names to preserve them through the GA process
        ga_kwargs['feature_names'] = feature_names

        hypotheses = hypothesis_search.generate_hypotheses(X, y, **ga_kwargs)
        gen_time = time.time() - start_gen_time
        logger.info(f"hypothesis_search.generate_hypotheses completed in {gen_time:.2f} seconds")

    hypothesis_time = time.time() - hypothesis_start
    logger.info(f"Hypothesis search phase completed in {hypothesis_time:.2f} seconds. Generated {len(hypotheses)} hypotheses")
    print(f"Generated {len(hypotheses)} hypotheses in {hypothesis_time:.2f} seconds")

    # Step 4: Statistical Analysis
    stat_start = time.time()
    print("Performing statistical analysis...")

    # Create feature name to index mapping
    feature_name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    logger.info(f"Feature mapping: {feature_name_to_index}")

    stat_results = []
    for i, hyp in enumerate(hypotheses):
        logger.info(f"Processing hypothesis {i+1}: features = {hyp['features']}")

        # Convert feature names to indices
        try:
            feature_indices = [feature_name_to_index[feat] for feat in hyp['features']]
            selected_features = X[:, feature_indices]
            logger.info(f"Mapped to indices: {feature_indices}")
        except KeyError as e:
            logger.error(f"Feature name not found in mapping: {e}")
            logger.error(f"Available features: {list(feature_name_to_index.keys())}")
            logger.error(f"Requested features: {hyp['features']}")
            raise

        result = statistical_analysis.perform_statistical_test(selected_features, y)
        stat_results.append(result)

    stat_time = time.time() - stat_start
    logger.info(f"Statistical analysis completed in {stat_time:.2f} seconds")
    print(f"Statistical analysis completed in {stat_time:.2f} seconds")

    # Step 4.5: Causal Analysis
    causal_start = time.time()
    print("Performing causal analysis...")
    causal_results = None
    if hypotheses:
        print(f"Number of hypotheses: {len(hypotheses)}")
        # Find the best hypothesis based on statistical significance
        p_values = [res.get('p_value', 1) for res in stat_results]
        print(f"P-values for hypotheses: {p_values}")
        best_hyp_idx = np.argmin(p_values)  # Lower p-value is better
        best_hyp = hypotheses[best_hyp_idx]
        print(f"Best hypothesis index: {best_hyp_idx}, features: {best_hyp['features']}")

        # Prepare data for causal analysis
        # Use the first selected feature as treatment, target as outcome, others as confounders
        if len(best_hyp['features']) > 0:
            treatment_feature = best_hyp['features'][0]  # Already a feature name
            outcome_col = target_col
            confounder_features = best_hyp['features'][1:]  # Already feature names
            print(f"Treatment: {treatment_feature}, Outcome: {outcome_col}, Confounders: {confounder_features}")

            # Create DataFrame with relevant columns
            causal_df = X_clean.copy()
            causal_df[outcome_col] = y
            print(f"Causal DataFrame shape: {causal_df.shape}")

            try:
                causal_results = causal_analysis.perform_causal_analysis(
                    causal_df,
                    treatment=treatment_feature,
                    outcome=outcome_col,
                    confounders=confounder_features
                )
                print(f"Causal analysis completed for hypothesis with treatment: {treatment_feature}")
                print(f"Causal results keys: {list(causal_results.keys()) if causal_results else 'None'}")
            except Exception as e:
                print(f"Causal analysis failed with exception: {e}")
                import traceback
                traceback.print_exc()
                causal_results = None
        else:
            print("Best hypothesis has no features, skipping causal analysis")
    else:
        print("No hypotheses generated, skipping causal analysis")
    causal_time = time.time() - causal_start
    print(f"Causal analysis completed in {causal_time:.2f} seconds")

    # Step 5: Narrative Generation
    narrative_start = time.time()
    print("Generating narratives...")
    summary_narrative = narrative_generation.generate_summary_narrative(hypotheses, feature_names, X, y, metadata={
        'data_shape': X.shape,
        'target_column': target_col,
        'feature_names': feature_names
    }, causal_results=causal_results)
    detailed_report = narrative_generation.create_report(hypotheses, feature_names, stat_results)
    narrative_time = time.time() - narrative_start
    print(f"Narrative generation completed in {narrative_time:.2f} seconds")

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
        'causal_results': causal_results,
        'summary_narrative': summary_narrative,
        'detailed_report': detailed_report
    }

    return results