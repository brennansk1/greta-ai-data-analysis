"""
Summary Narratives Module

Contains functions for generating summary narratives and reports from analysis results.
"""

from typing import Dict, List, Any
import numpy as np
from .regression_narratives import generate_regression_narrative
from .time_series_narratives import generate_time_series_narrative
from .causal_narratives import generate_causal_narrative, generate_causal_insight


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
    # hypothesis['features'] now contains feature names directly, not indices
    features = hypothesis['features']
    significance = hypothesis['significance']
    effect_size = hypothesis['effect_size']
    coverage = hypothesis['coverage']
    analysis_type = hypothesis.get('analysis_type', 'basic')

    # Determine confidence level
    if significance >= 0.95:
        confidence = "very high"
    elif significance >= 0.90:
        confidence = "high"
    elif significance >= 0.80:
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
        coverage_desc = "explains most of what's happening"
    elif coverage > 0.5:
        coverage_desc = "explains a good deal of what's happening"
    elif coverage > 0.3:
        coverage_desc = "explains some of what's happening"
    else:
        coverage_desc = "explains only a little of what's happening"

    if analysis_type == 'regression' and data is not None and target is not None:
        narrative = generate_regression_narrative(hypothesis, feature_names, data, target, confidence, effect, coverage_desc)
    elif analysis_type == 'time_series' and target is not None:
        narrative = generate_time_series_narrative(hypothesis, feature_names, data, target, confidence, effect, coverage_desc)
    else:  # basic or fallback
        if len(features) == 0:
            narrative = "No specific features were identified in this hypothesis. "
        elif len(features) == 1:
            narrative = f"The analysis suggests that {features[0]} has a {effect} relationship with the target variable. "
        else:
            feature_list = ", ".join(features[:-1]) + f" and {features[-1]}"
            narrative = f"The combination of {feature_list} shows a {effect} relationship with the target variable. "

        narrative += f"We're {confidence} about this finding and it {coverage_desc} in the data."

    return narrative


def generate_summary_narrative(hypotheses: List[Dict[str, Any]], feature_names: List[str], data: np.ndarray = None, target: np.ndarray = None, metadata: Dict[str, Any] = None, causal_results: Dict[str, Any] = None) -> str:
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
    methodology = "GRETA used smart computer algorithms to try out different combinations of your data features, testing each idea with careful statistical checks to see what really matters. "

    # Main findings
    findings = f"The analysis identified {num_hypotheses} potential {'hypothesis' if num_hypotheses == 1 else 'hypotheses'}. "

    if num_hypotheses == 1:
        findings += "The top finding is: "
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

        overall = f" Overall, we're {confidence_level} sure about these hypotheses, which have {effect_strength} effects and explain {coverage_desc} of what's happening in the data."

        # Interpretation guidance
        interpretation = " What this means: "
        if avg_significance > 0.9 and avg_effect > 0.4:
            interpretation += "These are solid findings you can trust. You might want to build models or use these insights to make better decisions."
        elif avg_significance > 0.8:
            interpretation += "These look promising, but check them with more data to be sure. Adding more examples or related factors could make them even stronger."
        else:
            interpretation += "These are early ideas that need more work. Try getting more data or including other variables to get a clearer picture."
    else:
        overall = ""
        interpretation = ""

    # Add causal analysis if available
    causal_narrative = ""
    if causal_results:
        causal_narrative = "\n\nCausal Analysis:\n" + generate_causal_narrative(causal_results)
        causal_insight = generate_causal_insight(causal_results)
        if causal_insight:
            causal_narrative += "\n" + causal_insight

    return data_overview + methodology + findings + overall + interpretation + causal_narrative


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
        significance_desc = "very unlikely to be due to chance"
        significance_note = "statistically significant"
    elif p_value < 0.10:
        significance_desc = "somewhat unlikely to be due to chance"
        significance_note = "marginally significant"
    else:
        significance_desc = "could easily be due to chance"
        significance_note = "not statistically significant"

    if r_squared > 0.5:
        practical_desc = "This relationship has practical importance - it matters a lot."
    elif r_squared > 0.2:
        practical_desc = "This is a noticeable relationship that could be useful."
    else:
        practical_desc = "This relationship might not be very important in practice."

    insight = f"This result is {significance_desc}. "
    insight += f"The model explains {int(r_squared * 100)}% of what's happening in the data. "
    insight += practical_desc
    insight += f" This result is {significance_note}."

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

    # Data Summary
    report += "Data Summary\n"
    if data is not None:
        report += f"- Dataset shape: {data.shape}\n"
    report += f"- Number of features: {len(feature_names)}\n"
    report += f"- Number of hypotheses generated: {len(hypotheses)}\n\n"

    report += generate_summary_narrative(hypotheses, feature_names, data, target) + "\n\n"

    report += "Detailed Findings:\n"
    for i, (hyp, stat) in enumerate(zip(hypotheses[:3], stat_results[:3]), 1):  # Top 3
        # Pass feature_names but the function now handles feature names directly
        report += f"{i}. {generate_hypothesis_narrative(hyp, feature_names, data, target)}\n"
        report += f"   {generate_insight_narrative(stat, hyp)}\n\n"

    report += "Recommendations:\n"
    if hypotheses and hypotheses[0]['significance'] > 0.95:
        report += "- The best finding is highly reliable and should be prioritized - focus on this one first.\n"
    elif hypotheses and hypotheses[0]['significance'] > 0.9:
        report += "- The best finding is very trustworthy and should be prioritized - focus on this one first.\n"
    if len(hypotheses) > 1:
        report += "- Look at mixing features from different ideas.\n"
    report += "- Check these results with more data or different tests to be sure."

    return report