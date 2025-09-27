"""
Output formatting and reporting utilities.
"""

import json
import yaml
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from fpdf import FPDF


def format_results(results: Dict[str, Any], format_type: str = 'json') -> str:
    """
    Format analysis results for output.

    Args:
        results: Analysis results dictionary.
        format_type: Output format ('json' or 'yaml').

    Returns:
        Formatted results string.
    """
    if format_type == 'json':
        return json.dumps(results, indent=2, default=str)
    elif format_type == 'yaml':
        return yaml.dump(results, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def generate_report(results: Dict[str, Any], format_type: str = 'text') -> str:
    """
    Generate a human-readable report from analysis results.

    Args:
        results: Analysis results dictionary.
        format_type: Report format ('text', 'markdown', 'html').

    Returns:
        Formatted report string.
    """
    metadata = results['metadata']
    hypotheses = results['hypotheses']
    summary_narrative = results['summary_narrative']
    detailed_report = results['detailed_report']
    causal_results = results.get('causal_results')

    if format_type == 'text':
        causal_section = ""
        if causal_results:
            from greta_core.causal_narratives import generate_causal_narrative
            causal_section = f"""

Causal Analysis
---------------
{generate_causal_narrative(causal_results)}
"""

        report = f"""
GRETA Analysis Report
=====================

Data Summary
------------
- Dataset shape: {metadata['data_shape']}
- Target column: {metadata['target_column']}
- Number of features: {len(metadata['feature_names'])}
- Number of hypotheses generated: {metadata['num_hypotheses']}

Summary
-------
{summary_narrative}

Detailed Findings
-----------------
{detailed_report}{causal_section}

Configuration Used
------------------
Data source: {metadata['config']['data']['source']}
Preprocessing: {metadata['config']['preprocessing']}
Hypothesis search: {metadata['config']['hypothesis_search']}
"""
        return report.strip()

    elif format_type == 'markdown':
        causal_section = ""
        if causal_results:
            from greta_core.causal_narratives import generate_causal_narrative
            causal_section = f"""

## Causal Analysis

{generate_causal_narrative(causal_results)}
"""

        report = f"""# GRETA Analysis Report

## Data Summary

- **Dataset shape**: {metadata['data_shape']}
- **Target column**: `{metadata['target_column']}`
- **Number of features**: {len(metadata['feature_names'])}
- **Number of hypotheses generated**: {metadata['num_hypotheses']}

## Summary

{summary_narrative}

## Detailed Findings

{detailed_report}{causal_section}

## Configuration Used

### Data Source
- **Source**: `{metadata['config']['data']['source']}`
- **Type**: {metadata['config']['data']['type']}

### Preprocessing
- **Missing strategy**: {metadata['config']['preprocessing']['missing_strategy']}
- **Outlier method**: {metadata['config']['preprocessing']['outlier_method']}
- **Feature engineering**: {metadata['config']['preprocessing']['feature_engineering']}

### Hypothesis Search
- **Population size**: {metadata['config']['hypothesis_search']['pop_size']}
- **Generations**: {metadata['config']['hypothesis_search']['num_generations']}
- **Crossover probability**: {metadata['config']['hypothesis_search']['cx_prob']}
- **Mutation probability**: {metadata['config']['hypothesis_search']['mut_prob']}
"""
        return report

    elif format_type == 'html':
        causal_section = ""
        if causal_results:
            from greta_core.causal_narratives import generate_causal_narrative
            causal_section = f"""
    <h2>Causal Analysis</h2>
    <div class="summary">
        {generate_causal_narrative(causal_results).replace(chr(10), '<br>')}
    </div>
"""

        report = f"""<!DOCTYPE html>
<html>
<head>
    <title>GRETA Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .config {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; }}
        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>GRETA Analysis Report</h1>

    <h2>Data Summary</h2>
    <ul>
        <li><strong>Dataset shape:</strong> {metadata['data_shape']}</li>
        <li><strong>Target column:</strong> {metadata['target_column']}</li>
        <li><strong>Number of features:</strong> {len(metadata['feature_names'])}</li>
        <li><strong>Number of hypotheses generated:</strong> {metadata['num_hypotheses']}</li>
    </ul>

    <h2>Summary</h2>
    <div class="summary">
        {summary_narrative.replace(chr(10), '<br>')}
    </div>

    <h2>Detailed Findings</h2>
    <pre>{detailed_report}</pre>{causal_section}

    <h2>Configuration Used</h2>
    <div class="config">
        <h3>Data Source</h3>
        <ul>
            <li><strong>Source:</strong> {metadata['config']['data']['source']}</li>
            <li><strong>Type:</strong> {metadata['config']['data']['type']}</li>
        </ul>

        <h3>Preprocessing</h3>
        <ul>
            <li><strong>Missing strategy:</strong> {metadata['config']['preprocessing']['missing_strategy']}</li>
            <li><strong>Outlier method:</strong> {metadata['config']['preprocessing']['outlier_method']}</li>
            <li><strong>Feature engineering:</strong> {metadata['config']['preprocessing']['feature_engineering']}</li>
        </ul>

        <h3>Hypothesis Search</h3>
        <ul>
            <li><strong>Population size:</strong> {metadata['config']['hypothesis_search']['pop_size']}</li>
            <li><strong>Generations:</strong> {metadata['config']['hypothesis_search']['num_generations']}</li>
            <li><strong>Crossover probability:</strong> {metadata['config']['hypothesis_search']['cx_prob']}</li>
            <li><strong>Mutation probability:</strong> {metadata['config']['hypothesis_search']['mut_prob']}</li>
        </ul>
    </div>
</body>
</html>"""
        return report

    elif format_type == 'pdf':
        pdf = FPDF()
        pdf.set_font("Helvetica", size=12)
        pdf.add_page()
        pdf.cell(200, 10, "GRETA Analysis Report", ln=True, align='C')
        pdf.ln(10)

        # Data Summary
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(200, 12, "What We Analyzed", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(200, 8, f"- Dataset with {metadata['data_shape'][0]:,} rows and {metadata['data_shape'][1]} columns", ln=True)
        pdf.cell(200, 8, f"- Predicting: {metadata['target_column']}", ln=True)
        pdf.cell(200, 8, f"- Features considered: {len(metadata['feature_names'])}", ln=True)
        pdf.cell(200, 8, f"- Ideas tested: {metadata['num_hypotheses']}", ln=True)
        pdf.ln(5)

        # Enhanced Features Used
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(200, 10, "Enhanced Pipeline Features Applied:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(200, 6, "* Parallel Execution - Multi-process genetic algorithm", ln=True)
        pdf.cell(200, 6, "* Dynamic Feature Engineering - Auto polynomial & interaction terms", ln=True)
        pdf.cell(200, 6, "* Importance Explainability - SHAP/permutation importance rankings", ln=True)
        pdf.cell(200, 6, "* Stability Selection - Bootstrap validation for robustness", ln=True)
        pdf.cell(200, 6, "* Causal Prioritization - Causal relationship weighting", ln=True)
        pdf.cell(200, 6, "* Adaptive Parameters - Dynamic GA parameter adjustment", ln=True)
        pdf.cell(200, 6, "* Multi-modal Handling - Advanced categorical encoding", ln=True)
        pdf.ln(10)

        # Summary
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(200, 12, "Key Findings", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(200, 8, summary_narrative.replace('\n', ' '))
        pdf.ln(10)

        # Top Hypotheses
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(200, 12, "Detailed Results", ln=True)
        pdf.ln(5)

        for i, hyp in enumerate(hypotheses, 1):
            # hyp['features'] now contains feature names directly
            feature_names = hyp['features']
            analysis_type = hyp.get('analysis_type', 'basic')
            significance = hyp['significance']
            effect_size = hyp['effect_size']
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(200, 10, f"Idea {i}", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(200, 8, f"- Type: {analysis_type}", ln=True)
            pdf.cell(200, 8, f"- Features: {', '.join(feature_names)}", ln=True)
            pdf.cell(200, 8, f"- How sure we are: {significance:.1%}", ln=True)
            pdf.cell(200, 8, f"- Impact: {'Powerful' if effect_size > 0.5 else 'Noticeable' if effect_size > 0.3 else 'Subtle'}", ln=True)
            pdf.ln(5)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        pdf.output(temp_file.name)
        with open(temp_file.name, 'rb') as f:
            content = f.read()
        os.unlink(temp_file.name)
        return content

    else:
        raise ValueError(f"Unsupported report format: {format_type}")


def save_output(content: str | bytes, file_path: Optional[str] = None) -> None:
    """
    Save content to file or print to stdout.

    Args:
        content: Content to save.
        file_path: File path to save to (None for stdout).
    """
    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        mode = 'wb' if isinstance(content, bytes) else 'w'
        encoding = None if isinstance(content, bytes) else 'utf-8'
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        print(f"Output saved to: {file_path}")
    else:
        if isinstance(content, bytes):
            print("Binary content cannot be printed to stdout.")
        else:
            print(content)