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

from greta_core import ingestion, preprocessing, hypothesis_search, statistical_analysis, narratives
from greta_core.statistical_analysis import causal_analysis
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
        load_start = time.time()
        try:
            if config.data.type == 'csv':
                df = ingestion.load_csv(config.data.source)
            elif config.data.type == 'excel':
                df = ingestion.load_excel(config.data.source, sheet_name=config.data.sheet_name)
            else:
                logger.error(f"Unsupported data type: {config.data.type}")
                raise ValueError(f"Unsupported data type: {config.data.type}")
            load_time = time.time() - load_start
            logger.info(f"Data loaded successfully in {load_time:.2f} seconds. Shape: {df.shape}, columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

        # Validate data
        logger.debug("Validating loaded data")
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
            logger.info(f"No target column specified, auto-selecting: {target_col}")
            print(f"No target column specified, using: {target_col}")
        else:
            target_col = config.data.target_column
            if target_col not in df.columns:
                logger.error(f"Specified target column '{target_col}' not found in data columns: {list(df.columns)}")
                raise ValueError(f"Target column '{target_col}' not found in data")
            logger.info(f"Using specified target column: {target_col}")

        # Separate features and target
        logger.info("Separating features and target")
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])
        logger.debug(f"Target shape: {y.shape}, Features shape: {X_df.shape}")

        # Encode categorical target if necessary
        if not np.issubdtype(y.dtype, np.number):
            logger.info("Encoding categorical target variable")
            unique_vals = np.unique(y)
            logger.debug(f"Target unique values: {unique_vals}")
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
        else:
            logger.debug("Target is already numeric")

        # Select only numeric columns for features
        logger.info("Selecting numeric feature columns")
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = [col for col in X_df.columns if col not in numeric_cols]
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric feature columns: {non_numeric_cols}")
        X_df = X_df[numeric_cols]
        feature_names = list(X_df.columns)
        logger.info(f"Selected {len(feature_names)} numeric features: {feature_names}")
        pbar.update(1)
    ingestion_time = time.time() - start_time
    logger.info(f"Data ingestion phase completed in {ingestion_time:.2f} seconds")
    print(f"Data Ingestion completed in {ingestion_time:.2f} seconds")

    # Step 2: Preprocessing
    logger.info("Starting preprocessing phase")
    preprocessing_start = time.time()
    with tqdm(total=4, desc="Preprocessing") as main_pbar:
        # Sub-stage 1: Data Profiling
        profile_start = time.time()
        logger.info("Starting data profiling")
        with tqdm(total=1, desc="Data Profiling", leave=False) as sub_pbar:
            profile = preprocessing.profile_data(X_df, progress_callback=lambda: sub_pbar.update(1))
            profile_time = time.time() - profile_start
            logger.info(f"Data profiling completed in {profile_time:.2f} seconds. Shape: {profile['shape']}, dtypes: {profile.get('dtypes', {})}")
            print(f"Data shape: {profile['shape']}")
        main_pbar.update(1)

        # Sub-stage 2: Missing Value Handling
        missing_start = time.time()
        logger.info(f"Starting missing value handling with strategy: {config.preprocessing.missing_strategy}")
        with tqdm(total=1, desc="Missing Value Handling", leave=False) as sub_pbar:
            X_clean = preprocessing.handle_missing_values(
                X_df,
                strategy=config.preprocessing.missing_strategy,
                progress_callback=lambda: sub_pbar.update(1)
            )
            missing_time = time.time() - missing_start
            logger.info(f"Missing value handling completed in {missing_time:.2f} seconds. Shape after: {X_clean.shape}")
        main_pbar.update(1)

        # Sub-stage 3: Outlier Detection
        outlier_start = time.time()
        logger.info(f"Starting outlier detection with method: {config.preprocessing.outlier_method}, threshold: {config.preprocessing.outlier_threshold}")
        with tqdm(total=1, desc="Outlier Detection", leave=False) as sub_pbar:
            outliers = preprocessing.detect_outliers(
                X_clean,
                method=config.preprocessing.outlier_method,
                threshold=config.preprocessing.outlier_threshold,
                progress_callback=lambda: sub_pbar.update(1)
            )

            # Remove outliers if any detected
            outlier_count = sum(len(indices) for indices in outliers.values())
            if outlier_count > 0:
                logger.info(f"Detected {outlier_count} outlier rows, removing them")
                X_clean = preprocessing.remove_outliers(X_clean, outliers)
                y = y[~np.isin(np.arange(len(y)), list(set().union(*outliers.values())))]
                print(f"Removed {outlier_count} outlier rows")
            else:
                logger.info("No outliers detected")
        outlier_time = time.time() - outlier_start
        logger.info(f"Outlier detection completed in {outlier_time:.2f} seconds. Final shape: {X_clean.shape}")
        main_pbar.update(1)

        # Sub-stage 4: Advanced Feature Engineering
        feature_start = time.time()
        logger.info("Starting advanced feature engineering")
        with tqdm(total=1, desc="Feature Engineering", leave=False) as sub_pbar:
            # Normalize data types
            if config.preprocessing.normalize_types:
                logger.info("Normalizing data types")
                X_clean = preprocessing.normalize_data_types(X_clean, progress_callback=lambda: sub_pbar.update(0.15) if sub_pbar.n < 1 else None)

            # Enhanced feature engineering pipeline
            if config.preprocessing.feature_engineering:
                logger.info("Performing comprehensive feature engineering")

                try:
                    # Use config parameters for feature engineering
                    encoding_method = config.preprocessing.encoding_method
                    numeric_transforms = config.preprocessing.numeric_transforms
                    max_interactions = config.preprocessing.max_interactions
                    max_cardinality = config.preprocessing.max_cardinality

                    # Enhanced feature engineering with new capabilities
                    X_clean, engineering_metadata = preprocessing.prepare_features_for_modeling(
                        X_clean, y, encoding_method=encoding_method,
                        numeric_transforms=numeric_transforms,
                        max_interactions=max_interactions, max_cardinality=max_cardinality,
                        progress_callback=lambda: sub_pbar.update(0.3) if sub_pbar.n < 1 else None
                    )

                    if engineering_metadata.get('success', True):
                        logger.info(f"Feature engineering completed. Original shape: {engineering_metadata['original_shape']}, Final shape: {engineering_metadata['final_shape']}")
                        logger.info(f"Generated {engineering_metadata.get('engineering_info', {}).get('total_features_generated', 0)} new features")
                    else:
                        logger.warning(f"Feature engineering completed with warnings: {engineering_metadata.get('warnings', [])}")

                except Exception as e:
                    logger.error(f"Feature engineering failed: {e}. Continuing with original features.")
                    # Continue with original X_clean

            # Feature selection
            if config.preprocessing.feature_selection:
                logger.info(f"Performing feature selection using {config.preprocessing.selection_method}")

                try:
                    # Apply feature selection
                    X_selected, selected_features, selection_info = preprocessing.select_features(
                        X_clean, y, method=config.preprocessing.selection_method,
                        k=config.preprocessing.max_features
                    )

                    logger.info(f"Feature selection completed. Selected {len(selected_features)} out of {selection_info['original_features']} features")
                    X_clean = X_selected if isinstance(X_clean, np.ndarray) else X_clean[selected_features]
                    feature_names = selected_features  # Update feature names

                except Exception as e:
                    logger.error(f"Feature selection failed: {e}. Continuing with all features.")
                    # Continue with current X_clean

            # Ensure progress bar completes
            if sub_pbar.n < 1:
                sub_pbar.update(1 - sub_pbar.n)

        feature_time = time.time() - feature_start
        logger.info(f"Advanced feature engineering completed in {feature_time:.2f} seconds")

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

        try:
            hypotheses = hypothesis_search.generate_hypotheses(X, y, **ga_kwargs)
            gen_time = time.time() - start_gen_time
            logger.info(f"hypothesis_search.generate_hypotheses completed successfully in {gen_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during hypothesis generation: {e}", exc_info=True)
            raise

    hypothesis_time = time.time() - hypothesis_start
    logger.info(f"Hypothesis search phase completed in {hypothesis_time:.2f} seconds. Generated {len(hypotheses)} hypotheses")
    if hypotheses:
        logger.debug(f"Sample hypothesis features: {hypotheses[0]['features'] if hypotheses else 'None'}")
    print(f"Generated {len(hypotheses)} hypotheses in {hypothesis_time:.2f} seconds")

    # Step 4: Statistical Analysis
    stat_start = time.time()
    logger.info("Starting statistical analysis phase")
    print("Performing statistical analysis...")

    # Create feature name to index mapping
    feature_name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    logger.debug(f"Feature name to index mapping created: {len(feature_name_to_index)} features")

    stat_results = []
    with tqdm(total=len(hypotheses), desc="Statistical Analysis") as pbar:
        for i, hyp in enumerate(hypotheses):
            logger.info(f"Processing hypothesis {i+1}/{len(hypotheses)}: features = {hyp['features']}")

            # Convert feature names to indices
            try:
                feature_indices = [feature_name_to_index[feat] for feat in hyp['features']]
                selected_features = X[:, feature_indices]
                logger.debug(f"Mapped to indices: {feature_indices}, selected shape: {selected_features.shape}")
            except KeyError as e:
                logger.error(f"Feature name not found in mapping: {e}")
                logger.error(f"Available features: {list(feature_name_to_index.keys())}")
                logger.error(f"Requested features: {hyp['features']}")
                raise

            try:
                result = statistical_analysis.perform_statistical_test(selected_features, y)
                logger.debug(f"Hypothesis {i+1} statistical result: p_value={result.get('p_value', 'N/A')}")
                stat_results.append(result)
            except Exception as e:
                logger.error(f"Error in statistical test for hypothesis {i+1}: {e}", exc_info=True)
                raise
            pbar.update(1)

    stat_time = time.time() - stat_start
    logger.info(f"Statistical analysis completed in {stat_time:.2f} seconds for {len(hypotheses)} hypotheses")
    print(f"Statistical analysis completed in {stat_time:.2f} seconds")

    # Step 4.5: Causal Analysis
    causal_start = time.time()
    logger.info("Starting causal analysis phase")
    print("Performing causal analysis...")
    causal_results = None
    if hypotheses:
        logger.info(f"Number of hypotheses available: {len(hypotheses)}")
        # Find the best hypothesis based on statistical significance
        p_values = [res.get('p_value', 1) for res in stat_results]
        logger.debug(f"P-values for hypotheses: {p_values}")
        best_hyp_idx = np.argmin(p_values)  # Lower p-value is better
        best_hyp = hypotheses[best_hyp_idx]
        logger.info(f"Selected best hypothesis index: {best_hyp_idx}, features: {best_hyp['features']}")

        # Prepare data for causal analysis
        # Use the first selected feature as treatment, target as outcome, others as confounders
        if len(best_hyp['features']) > 0:
            treatment_feature = best_hyp['features'][0]  # Already a feature name
            outcome_col = target_col
            confounder_features = best_hyp['features'][1:]  # Already feature names
            logger.debug(f"Treatment: {treatment_feature}, Outcome: {outcome_col}, Confounders: {confounder_features}")

            # Create DataFrame with relevant columns
            causal_df = X_clean.copy()
            causal_df[outcome_col] = y
            logger.info(f"Causal DataFrame prepared with shape: {causal_df.shape}")

            try:
                causal_results = causal_analysis.perform_causal_analysis(
                    causal_df,
                    treatment=treatment_feature,
                    outcome=outcome_col,
                    confounders=confounder_features
                )
                logger.info(f"Causal analysis completed successfully for treatment: {treatment_feature}")
                logger.debug(f"Causal results keys: {list(causal_results.keys()) if causal_results else 'None'}")
                print(f"Causal analysis completed for hypothesis with treatment: {treatment_feature}")
            except Exception as e:
                logger.error(f"Causal analysis failed: {e}", exc_info=True)
                print(f"Causal analysis failed with exception: {e}")
                causal_results = None
        else:
            logger.warning("Best hypothesis has no features, skipping causal analysis")
            print("Best hypothesis has no features, skipping causal analysis")
    else:
        logger.warning("No hypotheses generated, skipping causal analysis")
        print("No hypotheses generated, skipping causal analysis")
    causal_time = time.time() - causal_start
    logger.info(f"Causal analysis phase completed in {causal_time:.2f} seconds")
    print(f"Causal analysis completed in {causal_time:.2f} seconds")

    # Step 5: Narrative Generation
    narrative_start = time.time()
    logger.info("Starting narrative generation phase")
    print("Generating narratives...")
    try:
        summary_narrative = narratives.generate_summary_narrative(hypotheses, feature_names, X, y, metadata={
            'data_shape': X.shape,
            'target_column': target_col,
            'feature_names': feature_names
        }, causal_results=causal_results)
        logger.debug("Summary narrative generated")
    except Exception as e:
        logger.error(f"Error generating summary narrative: {e}", exc_info=True)
        summary_narrative = "Error generating summary narrative."

    try:
        detailed_report = narratives.create_report(hypotheses, feature_names, stat_results)
        logger.debug("Detailed report generated")
    except Exception as e:
        logger.error(f"Error generating detailed report: {e}", exc_info=True)
        detailed_report = "Error generating detailed report."

    narrative_time = time.time() - narrative_start
    logger.info(f"Narrative generation completed in {narrative_time:.2f} seconds")
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