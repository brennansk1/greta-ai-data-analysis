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
    import time
    start_time = time.time()
    logger.info("Starting Greta project initialization")
    logger.debug(f"Command arguments: output={output}, data_source={data_source}, data_type={data_type}, target_column={target_column}, interactive={interactive}")
    typer.echo("Initializing Greta project...")

    if interactive:
        logger.info("Running interactive wizard for configuration setup")
        # Interactive wizard
        if not data_source:
            data_source = typer.prompt("Path to data file", type=str)
            logger.debug(f"User provided data_source: {data_source}")
        data_type = typer.prompt("Data file type", default=data_type, type=str)
        logger.debug(f"Data type set to: {data_type}")
        target_column_input = typer.prompt("Target column name (leave empty to auto-detect)", default="", type=str)
        if target_column_input:
            target_column = target_column_input
            logger.debug(f"Target column set to: {target_column}")

    if not data_source:
        logger.error("Data source is required but not provided. Exiting initialization.")
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
    logger.debug("Default GretaConfig instance created")

    # Save config
    try:
        save_config(config, output)
        logger.info(f"Config saved successfully to: {output}")
        typer.echo(f"Config saved to: {output}")
    except Exception as e:
        logger.error(f"Failed to save config to {output}: {e}", exc_info=True)
        typer.echo(f"Error saving config: {e}")
        raise typer.Exit(1)

    # Validate config
    try:
        logger.info("Loading and validating saved configuration")
        loaded_config = load_config(output)
        warnings = validate_config(loaded_config)
        if warnings:
            logger.warning(f"Configuration validation completed with warnings: {warnings}")
            typer.echo("Configuration warnings:")
            for warning in warnings:
                typer.echo(f"  - {warning}")
        else:
            logger.info("Configuration validation passed without warnings")
            typer.echo("Configuration is valid.")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        typer.echo(f"Configuration validation failed: {e}")
        raise typer.Exit(1)

    init_time = time.time() - start_time
    logger.info(f"Greta project initialization completed successfully in {init_time:.2f} seconds")


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
    import time
    total_start_time = time.time()
    logger.info("Starting Greta analysis pipeline execution")
    logger.debug(f"Command arguments: config={config}, output={output}, format={format}, override={override}")
    typer.echo("Starting Greta analysis...")

    # Load config
    config_load_start = time.time()
    try:
        logger.info(f"Loading configuration from {config}")
        greta_config = load_config(config)
        config_load_time = time.time() - config_load_start
        logger.info(f"Configuration loaded successfully in {config_load_time:.2f} seconds")
        logger.debug(f"Loaded config data source: {greta_config.data.source}, type: {greta_config.data.type}")
    except Exception as e:
        logger.error(f"Error loading config from {config}: {e}", exc_info=True)
        typer.echo(f"Error loading config: {e}")
        raise typer.Exit(1)

    # Parse overrides
    overrides = {}
    if override:
        try:
            logger.info(f"Parsing parameter overrides: {override}")
            overrides = json.loads(override)
            logger.info(f"Overrides parsed successfully: {len(overrides)} parameters overridden")
            logger.debug(f"Override keys: {list(overrides.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing overrides JSON: {e}", exc_info=True)
            typer.echo(f"Error parsing overrides: {e}")
            raise typer.Exit(1)

    # Run analysis
    analysis_start = time.time()
    try:
        logger.info("Starting analysis pipeline execution with core integration")
        results = run_analysis_pipeline(greta_config, overrides)
        analysis_time = time.time() - analysis_start
        logger.info(f"Analysis pipeline completed successfully in {analysis_time:.2f} seconds")
        logger.debug(f"Results contain {len(results)} top-level keys: {list(results.keys())}")
    except Exception as e:
        logger.error(f"Error during analysis pipeline execution: {e}", exc_info=True)
        typer.echo(f"Error during analysis: {e}")
        raise typer.Exit(1)

    # Format and output results
    output_start = time.time()
    logger.info(f"Formatting results as {format}")
    try:
        formatted_results = format_results(results, format)
        format_time = time.time() - output_start
        logger.info(f"Results formatted successfully in {format_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error formatting results: {e}", exc_info=True)
        typer.echo(f"Error formatting results: {e}")
        raise typer.Exit(1)

    if output:
        try:
            logger.info(f"Saving formatted results to {output}")
            save_output(formatted_results, output)
            logger.info(f"Results saved successfully to {output}")
        except Exception as e:
            logger.error(f"Error saving results to {output}: {e}", exc_info=True)
            typer.echo(f"Error saving results: {e}")
            raise typer.Exit(1)
    else:
        logger.info("Displaying formatted results to stdout")
        typer.echo(formatted_results)

    total_time = time.time() - total_start_time
    logger.info(f"CLI run command completed successfully in {total_time:.2f} seconds")
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
    import time
    report_start_time = time.time()
    logger.info("Starting report generation")
    logger.debug(f"Command arguments: input_file={input_file}, output={output}, format={format}")
    typer.echo("Generating report...")

    # Load results
    load_start = time.time()
    try:
        logger.info(f"Loading results from {input_file}")
        with open(input_file, 'r') as f:
            if input_file.endswith('.json'):
                results = json.load(f)
                logger.debug("Results loaded as JSON")
            else:
                import yaml
                results = yaml.safe_load(f)
                logger.debug("Results loaded as YAML")
        load_time = time.time() - load_start
        logger.info(f"Results loaded successfully in {load_time:.2f} seconds")
        logger.debug(f"Results metadata: shape={results.get('metadata', {}).get('data_shape')}, hypotheses={results.get('metadata', {}).get('num_hypotheses')}")
    except Exception as e:
        logger.error(f"Error loading results from {input_file}: {e}", exc_info=True)
        typer.echo(f"Error loading results: {e}")
        raise typer.Exit(1)

    # Generate report
    gen_start = time.time()
    try:
        logger.info(f"Generating report in {format} format")
        report_content = generate_report(results, format)
        gen_time = time.time() - gen_start
        logger.info(f"Report generated successfully in {gen_time:.2f} seconds")
        logger.debug(f"Report content length: {len(report_content)} characters")
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        typer.echo(f"Error generating report: {e}")
        raise typer.Exit(1)

    # Output report
    output_start = time.time()
    if output:
        try:
            logger.info(f"Saving report to {output}")
            save_output(report_content, output)
            output_time = time.time() - output_start
            logger.info(f"Report saved successfully in {output_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error saving report to {output}: {e}", exc_info=True)
            typer.echo(f"Error saving report: {e}")
            raise typer.Exit(1)
    else:
        logger.info("Displaying report to stdout")
        typer.echo(report_content)

    total_report_time = time.time() - report_start_time
    logger.info(f"Report generation command completed successfully in {total_report_time:.2f} seconds")


if __name__ == "__main__":
    app()