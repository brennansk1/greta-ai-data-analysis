"""
Output formatting and reporting utilities.
"""

import json
import yaml
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fpdf import FPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import matplotlib.pyplot as plt
import io
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def format_results(results: Dict[str, Any], format_type: str = 'json') -> str:
    """
    Format analysis results for output.

    Args:
        results: Analysis results dictionary.
        format_type: Output format ('json' or 'yaml').

    Returns:
        Formatted results string.
    """
    logger.debug(f"Formatting results in {format_type} format")
    try:
        if format_type == 'json':
            formatted = json.dumps(results, indent=2, default=str)
            logger.debug(f"Results formatted as JSON, length: {len(formatted)} characters")
            return formatted
        elif format_type == 'yaml':
            formatted = yaml.dump(results, default_flow_style=False, sort_keys=False)
            logger.debug(f"Results formatted as YAML, length: {len(formatted)} characters")
            return formatted
        else:
            logger.error(f"Unsupported format requested: {format_type}")
            raise ValueError(f"Unsupported format: {format_type}")
    except Exception as e:
        logger.error(f"Error formatting results: {e}", exc_info=True)
        raise


def generate_report(results: Dict[str, Any], format_type: str = 'text') -> str:
    """
    Generate a human-readable report from analysis results.

    Args:
        results: Analysis results dictionary.
        format_type: Report format ('text', 'markdown', 'html').

    Returns:
        Formatted report string.
    """
    logger.info(f"Generating report in {format_type} format")
    logger.debug(f"Results keys: {list(results.keys())}")
    metadata = results['metadata']
    hypotheses = results['hypotheses']
    summary_narrative = results['summary_narrative']
    detailed_report = results['detailed_report']
    causal_results = results.get('causal_results')
    logger.debug(f"Metadata: data_shape={metadata.get('data_shape')}, num_hypotheses={metadata.get('num_hypotheses')}")

    if format_type == 'text':
        causal_section = ""
        if causal_results:
            from greta_core.narratives import generate_causal_narrative
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
        logger.debug("Generated text format report")
        return report.strip()

    elif format_type == 'markdown':
        causal_section = ""
        if causal_results:
            from greta_core.narratives import generate_causal_narrative
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
        logger.debug("Generated markdown format report")
        return report

    elif format_type == 'html':
        causal_section = ""
        if causal_results:
            from greta_core.narratives import generate_causal_narrative
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
        logger.debug("Generated HTML format report")
        return report

    elif format_type == 'pdf':
        # Create comprehensive PDF report using ReportLab
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()

        doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
        )

        story = []

        # Title
        story.append(Paragraph("GRETA Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Analysis Date
        story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Data Summary Section
        story.append(Paragraph("Dataset Summary", section_style))
        summary_data = [
            ["Metric", "Value"],
            ["Dataset Shape", f"{metadata['data_shape'][0]:,} observations, {metadata['data_shape'][1]} features"],
            ["Target Column", metadata['target_column']],
            ["Number of Features", str(len(metadata['feature_names']))],
            ["Number of Hypotheses Generated", str(metadata['num_hypotheses'])],
        ]
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 12))

        # Key Findings
        story.append(Paragraph("Key Findings", section_style))
        findings = summary_narrative.split('\n')
        for finding in findings:
            if finding.strip():
                story.append(Paragraph(finding.strip(), styles['Normal']))
                story.append(Spacer(1, 6))

        # Hypotheses Table
        if hypotheses:
            story.append(Paragraph("Top Hypotheses", section_style))
            hypotheses_data = [
                ["Rank", "Features", "Significance", "Effect Size", "Coverage"],
            ]
            # Extract top hypotheses from results
            top_hypotheses = hypotheses[:10]  # Top 10
            for i, hyp in enumerate(top_hypotheses, 1):
                features = ', '.join(hyp.get('features', []))
                sig = f"{hyp.get('significance', 0):.4f}"
                effect = f"{hyp.get('effect_size', 0):.4f}"
                coverage = f"{hyp.get('coverage', 0):.4f}"
                hypotheses_data.append([str(i), features, sig, effect, coverage])

            hypotheses_table = Table(hypotheses_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 1*inch, 1*inch])
            hypotheses_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(hypotheses_table)
            story.append(Spacer(1, 12))

        # Statistical Results
        if 'statistical_results' in results and results['statistical_results']:
            story.append(Paragraph("Statistical Analysis Results", section_style))
            stat_data = [
                ["Hypothesis", "Test Type", "P-Value", "Effect Size", "Confidence"],
            ]
            for i, stat in enumerate(results['statistical_results'][:5], 1):  # Top 5
                test_type = stat.get('test_type', 'Unknown')
                p_value = f"{stat.get('p_value', 'N/A')}"
                effect_size = f"{stat.get('effect_size', 'N/A')}"
                confidence = f"{stat.get('confidence', 'N/A')}"
                stat_data.append([f"H{i}", test_type, p_value, effect_size, confidence])

            stat_table = Table(stat_data, colWidths=[1*inch, 1.5*inch, 1*inch, 1*inch, 1.5*inch])
            stat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(stat_table)
            story.append(Spacer(1, 12))

        # Causal Analysis
        if causal_results:
            story.append(Paragraph("Causal Analysis", section_style))
            from greta_core.narratives import generate_causal_narrative
            causal_narrative = generate_causal_narrative(causal_results)
            causal_lines = causal_narrative.split('\n')
            for line in causal_lines:
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['Normal']))
                    story.append(Spacer(1, 4))

            # Causal results table if available
            if 'estimation' in causal_results:
                story.append(Spacer(1, 12))
                story.append(Paragraph("Causal Effect Estimates", styles['Heading3']))
                effect_data = [
                    ["Method", "Estimate", "Confidence Interval", "P-Value"],
                ]
                est = causal_results['estimation']
                method = est.get('method', 'Unknown')
                estimate = f"{est.get('estimate', 'N/A')}"
                ci = f"[{est.get('ci_lower', 'N/A')}, {est.get('ci_upper', 'N/A')}]"
                p_val = f"{est.get('p_value', 'N/A')}"
                effect_data.append([method, estimate, ci, p_val])

                effect_table = Table(effect_data, colWidths=[1.5*inch, 1*inch, 2*inch, 1*inch])
                effect_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                ]))
                story.append(effect_table)
            story.append(Spacer(1, 12))

        # Feature Importance Visualization
        if hypotheses and len(hypotheses) > 0:
            story.append(Paragraph("Feature Importance Analysis", section_style))

            # Create feature importance data
            feature_importance = {}
            for hyp in hypotheses[:5]:  # Top 5 hypotheses
                for feature in hyp.get('features', []):
                    if feature in feature_importance:
                        feature_importance[feature] += hyp.get('significance', 0)
                    else:
                        feature_importance[feature] = hyp.get('significance', 0)

            if feature_importance:
                # Create bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                features = list(feature_importance.keys())[:8]  # Top 8 features
                importance = [feature_importance[f] for f in features]

                ax.barh(features, importance, color='skyblue')
                ax.set_xlabel('Importance Score')
                ax.set_title('Feature Importance (Top Features)')
                plt.tight_layout()

                # Save to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)

                # Add to PDF
                img = Image(buf)
                img.drawHeight = 3*inch
                img.drawWidth = 6*inch
                story.append(img)
                story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("Recommendations", section_style))
        recommendations = [
            "1. Focus retention strategies on the most significant features identified in the analysis.",
            "2. Implement A/B testing for the top hypotheses to validate findings in production.",
            "3. Monitor the key metrics identified to track changes over time.",
            "4. Consider collecting additional data for features with high importance scores.",
            "5. Use the causal analysis insights to understand true impact relationships.",
            "6. Regularly re-run the analysis as new data becomes available.",
        ]
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 6))

        # Configuration Used
        story.append(Paragraph("Analysis Configuration", section_style))
        config = metadata.get('config', {})
        config_items = [
            f"Data Source: {config.get('data', {}).get('source', 'N/A')}",
            f"Preprocessing: Missing Strategy - {config.get('preprocessing', {}).get('missing_strategy', 'N/A')}",
            f"Hypothesis Search: Population Size - {config.get('hypothesis_search', {}).get('pop_size', 'N/A')}",
            f"GA Generations: {config.get('hypothesis_search', {}).get('num_generations', 'N/A')}",
        ]
        for item in config_items:
            story.append(Paragraph(item, styles['Normal']))
            story.append(Spacer(1, 4))

        # Build PDF
        doc.build(story)

        with open(temp_file.name, 'rb') as f:
            content = f.read()
        os.unlink(temp_file.name)
        logger.debug("Generated comprehensive PDF format report")
        return content

    else:
        logger.error(f"Unsupported report format requested: {format_type}")
        raise ValueError(f"Unsupported report format: {format_type}")


def save_output(content: str | bytes, file_path: Optional[str] = None) -> None:
    """
    Save content to file or print to stdout.

    Args:
        content: Content to save.
        file_path: File path to save to (None for stdout).
    """
    logger.debug(f"Saving output: file_path={file_path}, content_type={type(content).__name__}, content_length={len(content)}")
    if file_path:
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            mode = 'wb' if isinstance(content, bytes) else 'w'
            encoding = None if isinstance(content, bytes) else 'utf-8'
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
            logger.info(f"Output saved successfully to: {file_path}")
            print(f"Output saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save output to {file_path}: {e}", exc_info=True)
            raise
    else:
        logger.debug("Outputting content to stdout")
        if isinstance(content, bytes):
            logger.warning("Binary content cannot be printed to stdout")
            print("Binary content cannot be printed to stdout.")
        else:
            print(content)