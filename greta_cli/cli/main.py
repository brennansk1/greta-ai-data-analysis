"""
Main CLI application using Typer.
"""

import typer
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import logging
import sys

from ..config import load_config, save_config, GretaConfig, DataConfig
from ..core_integration import run_analysis_pipeline
from ..output import format_results, generate_report, save_output
from ..config import load_config, save_config, GretaConfig, DataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('greta_cli.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def validate_config(config: GretaConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Args:
        config: GretaConfig instance to validate.

    Returns:
        List of warning/error messages.
    """
    warnings = []

    # Check data source exists
    if not Path(config.data.source).exists():
        warnings.append(f"Data source file does not exist: {config.data.source}")

    # Check hypothesis search parameters
    if config.hypothesis_search.pop_size < 10:
        warnings.append("Population size is very small, may not find good hypotheses")

    if config.hypothesis_search.num_generations < 5:
        warnings.append("Number of generations is low, may not converge")

    return warnings


app = typer.Typer(help="GRETA CLI - Automated Data Analysis with Genetic Algorithms")

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
    logger.info("Starting Greta project initialization")
    typer.echo("Initializing Greta project...")

    if interactive:
        logger.info("Running interactive wizard")
        # Interactive wizard
        if not data_source:
            data_source = typer.prompt("Path to data file", type=str)
        data_type = typer.prompt("Data file type", default=data_type, type=str)
        target_column_input = typer.prompt("Target column name (leave empty to auto-detect)", default="", type=str)
        if target_column_input:
            target_column = target_column_input

    if not data_source:
        logger.error("Data source is required but not provided")
        typer.echo("Error: Data source is required. Use --data-source or --interactive")
        raise typer.Exit(1)

    logger.info(f"Creating config with data_source={data_source}, data_type={data_type}, target_column={target_column}")

    # Create default config
    data_config = DataConfig(
        source=data_source,
        type=data_type,
        target_column=target_column
    )

    config = GretaConfig(data=data_config)

    # Save config
    save_config(config, output)
    logger.info(f"Config saved to: {output}")
    typer.echo(f"Config saved to: {output}")

    # Validate config
    try:
        loaded_config = load_config(output)
        warnings = validate_config(loaded_config)
        if warnings:
            logger.warning(f"Configuration warnings: {warnings}")
            typer.echo("Configuration warnings:")
            for warning in warnings:
                typer.echo(f"  - {warning}")
        else:
            logger.info("Configuration validation passed")
            typer.echo("Configuration is valid.")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
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
    logger.info("Starting Greta analysis pipeline")
    typer.echo("Starting Greta analysis...")

    # Load config
    try:
        logger.info(f"Loading configuration from {config}")
        greta_config = load_config(config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        typer.echo(f"Error loading config: {e}")
        raise typer.Exit(1)

    # Parse overrides
    overrides = {}
    if override:
        try:
            logger.info(f"Parsing parameter overrides: {override}")
            overrides = json.loads(override)
            logger.info(f"Overrides parsed successfully: {overrides}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing overrides: {e}")
            typer.echo(f"Error parsing overrides: {e}")
            raise typer.Exit(1)

    # Run analysis
    try:
        logger.info("Starting analysis pipeline execution")
        results = run_analysis_pipeline(greta_config, overrides)
        logger.info("Analysis pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        typer.echo(f"Error during analysis: {e}")
        raise typer.Exit(1)

    # Format and output results
    logger.info(f"Formatting results as {format}")
    formatted_results = format_results(results, format)

    if output:
        logger.info(f"Saving results to {output}")
        save_output(formatted_results, output)
    else:
        logger.info("Displaying results to stdout")
        typer.echo(formatted_results)

    logger.info("CLI run command completed successfully")
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