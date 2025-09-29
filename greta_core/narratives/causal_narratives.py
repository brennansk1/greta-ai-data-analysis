"""
Causal Narratives Module

Contains functions for generating narratives from causal analysis results using DoWhy.
"""

import numpy as np
from typing import Dict, Any, Optional


def generate_causal_narrative(causal_results: Dict[str, Any]) -> str:
    """
    Generate a plain-English narrative from causal analysis results.

    Args:
        causal_results: Dictionary containing causal analysis results.

    Returns:
        Plain-English description of causal findings.
    """
    try:
        if not causal_results or 'estimation' not in causal_results:
            return "No causal analysis was performed."

        estimation = causal_results['estimation']
        treatment = causal_results.get('treatment', 'treatment')
        outcome = causal_results.get('outcome', 'outcome')
        confounders = causal_results.get('confounders', [])

        estimate = estimation.get('estimate', 0)
        method = estimation.get('method', 'unknown')

        # Ensure estimate is numeric
        try:
            estimate = float(estimate)
        except (TypeError, ValueError):
            return f"Causal analysis was performed but the estimate value ({estimate}) is not numeric."

        # Interpret the causal effect
        if abs(estimate) < 0.01:
            effect_desc = "no significant causal effect"
        elif abs(estimate) < 0.1:
            effect_desc = "a small causal effect"
        elif abs(estimate) < 0.3:
            effect_desc = "a moderate causal effect"
        else:
            effect_desc = "a strong causal effect"

        direction = "increases" if estimate > 0 else "decreases"

        # Confidence interpretation
        confidence_intervals = estimation.get('confidence_intervals')
        if confidence_intervals is not None:
            try:
                # Handle different formats of confidence intervals
                if isinstance(confidence_intervals, np.ndarray) and confidence_intervals.ndim == 2:
                    ci_lower, ci_upper = confidence_intervals[0, 0], confidence_intervals[0, 1]
                elif isinstance(confidence_intervals, (list, tuple)) and len(confidence_intervals) >= 2:
                    ci_lower, ci_upper = confidence_intervals[0], confidence_intervals[1]
                else:
                    ci_lower, ci_upper = None, None
                if ci_lower is not None and ci_upper is not None and ci_lower * ci_upper > 0:  # Same sign, significant
                    confidence_desc = "we're confident that"
                else:
                    confidence_desc = "we're not entirely sure, but it suggests that"
            except Exception as e:
                confidence_desc = "it appears that"
        else:
            confidence_desc = "it appears that"

        # Confounders mention
        try:
            if confounders:
                confounder_text = f", even after accounting for {', '.join(str(c) for c in confounders)}"
            else:
                confounder_text = ""
        except Exception as e:
            confounder_text = ""

        narrative = f"Following enhanced preprocessing and feature engineering (including advanced encoding and automated feature generation), causal analysis using the {method} method suggests that {treatment} has {effect_desc} on {outcome}{confounder_text}. "
        narrative += f"{confidence_desc} each unit increase in {treatment} {direction} {outcome} by approximately {abs(estimate):.3f} units."

        # Add interpretation
        if abs(estimate) > 0.1:
            narrative += " This suggests a meaningful causal relationship that could be important for decision-making."
        else:
            narrative += " While statistically significant, the practical impact might be limited."

        return narrative
    except Exception as e:
        return f"Error generating causal narrative: {str(e)}"


def generate_causal_insight(causal_results: Dict[str, Any]) -> str:
    """
    Generate insights from causal analysis validation.

    Args:
        causal_results: Causal analysis results.

    Returns:
        Insightful narrative about causal assumptions.
    """
    if not causal_results or 'identification' not in causal_results:
        return ""

    identification = causal_results['identification']
    backdoor_vars = identification.get('backdoor_variables', [])
    instrumental_vars = identification.get('instrumental_variables', [])

    insight = ""

    if backdoor_vars:
        insight += f"The analysis identified {len(backdoor_vars)} confounding variables that were controlled for. "

    if instrumental_vars:
        insight += f"Potential instrumental variables were found that could strengthen the causal claim. "

    # Add general causal insight
    insight += "Remember that causal inference relies on certain assumptions - correlation doesn't always mean causation, but this analysis attempts to address that distinction."

    return insight