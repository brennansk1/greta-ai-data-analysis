"""
Narrative Generation Module

Translates statistical results into plain-English insights and explanations.
Creates human-readable summaries of findings, including confidence levels
and practical implications.
"""

from typing import Dict, List, Any
import numpy as np
from .statistical_analysis import perform_multiple_linear_regression, detect_trend, detect_seasonality


def generate_regression_narrative(hypothesis: Dict[str, Any], feature_names: List[str], data: np.ndarray, target: np.ndarray, confidence: str, effect: str, coverage_desc: str) -> str:
    """
    Generate narrative for regression analysis.

    Args:
        hypothesis: Hypothesis details.
        feature_names: Feature names.
        data: Feature matrix.
        target: Target variable.
        confidence: Confidence level string.
        effect: Effect strength string.
        coverage_desc: Coverage description.

    Returns:
        Regression narrative.
    """
    if data is None or target is None:
        # Fallback to basic narrative if data not available
        features = [feature_names[i] for i in hypothesis['features']]
        if len(features) == 1:
            narrative = f"The analysis suggests that {features[0]} has a {effect} relationship with the target variable. "
        else:
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            narrative = f"The combination of {feature_list} shows a {effect} relationship with the target variable. "
        narrative += f"This finding has {confidence} statistical confidence and {coverage_desc} in the data."
        return narrative

    features = [feature_names[i] for i in hypothesis['features']]
    selected_data = data[:, hypothesis['features']]

    # Perform regression to get detailed results
    reg_results = perform_multiple_linear_regression(selected_data, target)
    coefficients = reg_results['coefficients']
    p_values = reg_results['p_values']
    adj_r_squared = reg_results['adj_r_squared']

    # Identify significant predictors
    significant_features = [features[i] for i, p in enumerate(p_values) if p < 0.05]

    if len(features) == 1:
        coeff_desc = f"with a coefficient of {coefficients[0]:.3f}"
        narrative = f"Multiple linear regression shows that {features[0]} has a {effect} linear relationship with the target variable {coeff_desc}. "
    else:
        feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
        narrative = f"Multiple linear regression analysis of {feature_list} reveals a {effect} relationship with the target variable. "

        if significant_features:
            sig_list = ", ".join(significant_features[:-1]) + f" and {significant_features[-1]}" if len(significant_features) > 1 else significant_features[0]
            narrative += f"The most significant predictors are {sig_list}. "

    narrative += f"The model has {confidence} statistical confidence, explains {adj_r_squared:.1%} of the variance in the target variable, and {coverage_desc}."

    return narrative


def generate_time_series_narrative(hypothesis: Dict[str, Any], feature_names: List[str], data: np.ndarray, target: np.ndarray, confidence: str, effect: str, coverage_desc: str) -> str:
    """
    Generate narrative for time-series analysis.

    Args:
        hypothesis: Hypothesis details.
        feature_names: Feature names.
        data: Feature matrix.
        target: Target variable.
        confidence: Confidence level string.
        effect: Effect strength string.
        coverage_desc: Coverage description.

    Returns:
        Time-series narrative.
    """
    if target is None:
        # Fallback to basic narrative if target not available
        features = [feature_names[i] for i in hypothesis['features']]
        if len(features) == 1:
            narrative = f"The analysis suggests that {features[0]} has a {effect} relationship with the target variable. "
        else:
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            narrative = f"The combination of {feature_list} shows a {effect} relationship with the target variable. "
        narrative += f"This finding has {confidence} statistical confidence and {coverage_desc} in the data."
        return narrative

    features = [feature_names[i] for i in hypothesis['features']]

    if len(features) == 0:
        # Analyze trend in target
        trend_results = detect_trend(target)
        slope = trend_results['slope']
        trend_strength = trend_results['trend_strength']

        direction = "increasing" if slope > 0 else "decreasing"
        strength_desc = "strong" if trend_strength > 0.5 else "moderate" if trend_strength > 0.3 else "weak"

        narrative = f"Time-series analysis of the target variable shows a {strength_desc} {direction} trend over time. "
        narrative += f"The trend has {confidence} statistical confidence and {coverage_desc}."
    else:
        if data is None:
            # Fallback if data not available
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            verb = "has" if len(features) == 1 else "have"
            narrative = f"Analysis of trends over time suggests that {feature_list} {verb} a {effect} relationship with the target variable. "
            narrative += f"This finding has {confidence} statistical confidence and {coverage_desc}."
        else:
            # Regression on time-series predictors
            selected_data = data[:, hypothesis['features']]
            reg_results = perform_multiple_linear_regression(selected_data, target)
            adj_r_squared = reg_results['adj_r_squared']

            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            verb = "has" if len(features) == 1 else "have"
            narrative = f"Analysis of trends over time shows that {feature_list} {verb} a {effect} relationship with the target variable. "
            narrative += f"The model has {confidence} statistical confidence, explains {adj_r_squared:.1%} of the variance in the target variable, and {coverage_desc}."

            # Check for seasonality if possible
            if len(target) > 12:
                seasonal_results = detect_seasonality(target)
                if seasonal_results['seasonal_strength'] > 0.1:
                    narrative += f" Additionally, there appears to be seasonal patterns in the data."

    return narrative


def generate_hypothesis_narrative(hypothesis: Dict[str, Any], feature_names: List[str], data: np.ndarray = None, target: np.ndarray = None) -> str:
    """
    Generate a plain-English narrative for a single hypothesis.

    Args:
        hypothesis: Dictionary containing hypothesis details.
        feature_names: List of feature names.
        data: Optional feature matrix for detailed analysis.
        target: Optional target variable for detailed analysis.

    Returns:
        Plain-English description of the hypothesis.
    """
    features = [feature_names[i] for i in hypothesis['features']]
    significance = hypothesis['significance']
    effect_size = hypothesis['effect_size']
    coverage = hypothesis['coverage']
    analysis_type = hypothesis.get('analysis_type', 'basic')

    # Determine confidence level
    if significance > 0.95:
        confidence = "very high"
    elif significance > 0.90:
        confidence = "high"
    elif significance > 0.80:
        confidence = "moderate"
    else:
        confidence = "low"

    # Determine effect strength
    if effect_size > 0.5:
        effect = "strong"
    elif effect_size > 0.3:
        effect = "moderate"
    else:
        effect = "weak"

    # Determine coverage
    if coverage > 0.7:
        coverage_desc = "covers most of the variation"
    elif coverage > 0.5:
        coverage_desc = "covers a significant portion of the variation"
    elif coverage > 0.3:
        coverage_desc = "covers some of the variation"
    else:
        coverage_desc = "covers little of the variation"

    if analysis_type == 'regression' and data is not None and target is not None:
        narrative = generate_regression_narrative(hypothesis, feature_names, data, target, confidence, effect, coverage_desc)
    elif analysis_type == 'time_series' and target is not None:
        narrative = generate_time_series_narrative(hypothesis, feature_names, data, target, confidence, effect, coverage_desc)
    else:  # basic or fallback
        if len(features) == 1:
            narrative = f"The analysis suggests that {features[0]} has a {effect} relationship with the target variable. "
        else:
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}" if len(features) > 1 else features[0]
            narrative = f"The combination of {feature_list} shows a {effect} relationship with the target variable. "

        narrative += f"This finding has {confidence} statistical confidence and {coverage_desc} in the data."

    return narrative


def generate_summary_narrative(hypotheses: List[Dict[str, Any]], feature_names: List[str], data: np.ndarray = None, target: np.ndarray = None, metadata: Dict[str, Any] = None) -> str:
    """
    Generate a summary narrative for multiple hypotheses.

    Args:
        hypotheses: List of hypothesis dictionaries.
        feature_names: List of feature names.
        data: Optional feature matrix.
        target: Optional target variable.
        metadata: Optional metadata dictionary with analysis context.

    Returns:
        Summary narrative.
    """
    if not hypotheses:
        return "No significant hypotheses were found in the data."

    top_hypothesis = hypotheses[0]
    num_hypotheses = len(hypotheses)

    # Data overview
    data_overview = ""
    if metadata:
        data_shape = metadata.get('data_shape', (0, 0))
        target_col = metadata.get('target_column', 'target')
        data_overview = f"This analysis examined a dataset with {data_shape[0]:,} observations and {data_shape[1]} features, analyzing patterns in the '{target_col}' variable. "

    # Methodology explanation
    methodology = "Using a genetic algorithm approach, GRETA explored various combinations of features through evolutionary optimization, evaluating each hypothesis using statistical tests including multiple linear regression and time-series analysis. "

    # Main findings
    findings = f"The analysis identified {num_hypotheses} potential hypotheses. "

    if num_hypotheses == 1:
        findings += "The key finding is: "
    else:
        findings += f"The strongest hypothesis (using {len(top_hypothesis['features'])} features) suggests: "

    findings += generate_hypothesis_narrative(top_hypothesis, feature_names, data, target)

    # Overall assessment
    if num_hypotheses > 1:
        avg_significance = np.mean([h['significance'] for h in hypotheses])
        avg_effect = np.mean([h['effect_size'] for h in hypotheses])
        avg_coverage = np.mean([h['coverage'] for h in hypotheses])

        confidence_level = "very high" if avg_significance > 0.95 else "high" if avg_significance > 0.9 else "moderate" if avg_significance > 0.8 else "low"
        effect_strength = "strong" if avg_effect > 0.5 else "moderate" if avg_effect > 0.3 else "weak"
        coverage_desc = "substantial portions" if avg_coverage > 0.5 else "some" if avg_coverage > 0.3 else "limited portions"

        overall = f" Overall, the hypotheses demonstrate {confidence_level} statistical confidence with {effect_strength} effects, explaining {coverage_desc} of the variance in the data."

        # Interpretation guidance
        interpretation = " These results suggest "
        if avg_significance > 0.9 and avg_effect > 0.4:
            interpretation += "reliable patterns that warrant further investigation. Consider developing predictive models or applying these insights to improve decision-making processes."
        elif avg_significance > 0.8:
            interpretation += "promising leads that should be validated with additional data or cross-validation. Explore collecting more observations or including related features to strengthen these relationships."
        else:
            interpretation += "preliminary insights that may benefit from more comprehensive analysis. Consider increasing the dataset size or incorporating additional relevant variables for better understanding."
    else:
        overall = ""
        interpretation = ""

    return data_overview + methodology + findings + overall + interpretation


def generate_insight_narrative(stat_results: Dict[str, float], hypothesis: Dict[str, Any]) -> str:
    """
    Generate insights based on statistical results.

    Args:
        stat_results: Statistical test results.
        hypothesis: Hypothesis details.

    Returns:
        Insightful narrative.
    """
    p_value = stat_results.get('p_value', 1.0)
    r_squared = stat_results.get('r_squared', 0.0)

    if p_value < 0.05:
        significance_desc = "statistically significant"
    elif p_value < 0.10:
        significance_desc = "marginally significant"
    else:
        significance_desc = "not statistically significant"

    if r_squared > 0.5:
        practical_desc = "This relationship has strong practical importance."
    elif r_squared > 0.2:
        practical_desc = "This relationship has moderate practical importance."
    else:
        practical_desc = "This relationship may have limited practical importance."

    insight = f"The statistical test shows the result is {significance_desc} (p = {p_value:.3f}). "
    insight += f"The model explains {r_squared:.1%} of the variance in the data. "
    insight += practical_desc

    if hypothesis['effect_size'] > 0.3:
        insight += " The effect size indicates this is not just a statistical artifact."

    return insight


def create_report(hypotheses: List[Dict[str, Any]], feature_names: List[str], stat_results: List[Dict[str, float]], data: np.ndarray = None, target: np.ndarray = None) -> str:
    """
    Create a complete analysis report.

    Args:
        hypotheses: List of hypotheses.
        feature_names: Feature names.
        stat_results: Statistical results for each hypothesis.
        data: Optional feature matrix.
        target: Optional target variable.

    Returns:
        Complete report narrative.
    """
    report = "GRETA Analysis Report\n\n"

    report += generate_summary_narrative(hypotheses, feature_names, data, target) + "\n\n"

    report += "Detailed Findings:\n"
    for i, (hyp, stat) in enumerate(zip(hypotheses[:3], stat_results[:3]), 1):  # Top 3
        report += f"{i}. {generate_hypothesis_narrative(hyp, feature_names, data, target)}\n"
        report += f"   {generate_insight_narrative(stat, hyp)}\n\n"

    report += "Recommendations:\n"
    if hypotheses and hypotheses[0]['significance'] > 0.9:
        report += "- The top hypothesis is highly reliable and should be prioritized for further investigation.\n"
    if len(hypotheses) > 1:
        report += "- Consider exploring combinations of features from multiple hypotheses.\n"
    report += "- Validate these findings with additional data or cross-validation techniques."

    return report