"""
Main CLI application using Typer.
"""

import typer
from pathlib import Path
from typing import Optional, Dict, Any
import json

from ..config import load_config, save_config, GretaConfig, DataConfig
from ..core_integration import run_analysis_pipeline
from ..output import format_results, generate_report, save_output

app = typer.Typer(help="GRETA CLI - Automated Data Analysis with Genetic Algorithms")


@app.command()
def init(
    output: str = typer.Option("config.yml", help="Path to save config file"),
    data_source: Optional[str] = typer.Option(None, help="Path to data file"),
    data_type: str = typer.Option("csv", help="Data file type (csv, excel)"),
    target_column: Optional[str] = typer.Option(None, help="Name of target column"),
    interactive: bool = typer.Option(False, help="Run interactive wizard")
):
    """
    Initialize a new Greta project by creating a config.yml file.
    """
    typer.echo("Initializing Greta project...")

    if interactive:
        # Interactive wizard
        if not data_source:
            data_source = typer.prompt("Path to data file", type=str)
        data_type = typer.prompt("Data file type", default=data_type, type=str)
        target_column_input = typer.prompt("Target column name (leave empty to auto-detect)", default="", type=str)
        if target_column_input:
            target_column = target_column_input

    if not data_source:
        typer.echo("Error: Data source is required. Use --data-source or --interactive")
        raise typer.Exit(1)

    # Create default config
    data_config = DataConfig(
        source=data_source,
        type=data_type,
        target_column=target_column
    )

    config = GretaConfig(data=data_config)

    # Save config
    save_config(config, output)
    typer.echo(f"Config saved to: {output}")

    # Validate config
    try:
        loaded_config = load_config(output)
        warnings = []  # load_config already validates via pydantic
        typer.echo("Configuration is valid.")
    except Exception as e:
        typer.echo(f"Configuration validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    config: str = typer.Option("config.yml", help="Path to config file"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
    format: str = typer.Option("json", help="Output format (json, yaml)"),
    override: Optional[str] = typer.Option(None, help="JSON string with parameter overrides")
):
    """
    Execute the analysis pipeline using the provided configuration.
    """
    typer.echo("Starting Greta analysis...")

    # Load config
    try:
        greta_config = load_config(config)
    except Exception as e:
        typer.echo(f"Error loading config: {e}")
        raise typer.Exit(1)

    # Parse overrides
    overrides = {}
    if override:
        try:
            overrides = json.loads(override)
        except json.JSONDecodeError as e:
            typer.echo(f"Error parsing overrides: {e}")
            raise typer.Exit(1)

    # Run analysis
    try:
        results = run_analysis_pipeline(greta_config, overrides)
    except Exception as e:
        typer.echo(f"Error during analysis: {e}")
        raise typer.Exit(1)

    # Format and output results
    formatted_results = format_results(results, format)

    if output:
        save_output(formatted_results, output)
    else:
        typer.echo(formatted_results)

    typer.echo("Analysis completed successfully.")


@app.command()
def report(
    input_file: str = typer.Option(..., help="Path to JSON results file"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
    format: str = typer.Option("text", help="Report format (text, markdown, html, pdf)")
):
    """
    Generate a human-readable report from analysis results.
    """
    typer.echo("Generating report...")

    # Load results
    try:
        with open(input_file, 'r') as f:
            if input_file.endswith('.json'):
                results = json.load(f)
            else:
                import yaml
                results = yaml.safe_load(f)
    except Exception as e:
        typer.echo(f"Error loading results: {e}")
        raise typer.Exit(1)

    # Generate report
    try:
        report_content = generate_report(results, format)
    except Exception as e:
        typer.echo(f"Error generating report: {e}")
        raise typer.Exit(1)

    # Output report
    if output:
        save_output(report_content, output)
    else:
        typer.echo(report_content)


if __name__ == "__main__":
    app()