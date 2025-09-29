# Greta CLI Architecture

## Overview

The Greta CLI (`greta-cli`) is the command-line interface component of Project Greta, designed for Phase 1 MVP. It provides a professional, scriptable interface for experts to leverage the Greta Core Engine for automated data analysis using genetic algorithms. The CLI emphasizes headless operation, configuration-driven workflows, and structured output to support integration into professional data pipelines.

## Overall Structure

The greta-cli is structured as a modular Python package with the following high-level organization:

- **Entry Point**: `greta` command, implemented via a Typer application.
- **Core Modules**:
  - `cli/`: Command definitions and argument parsing.
  - `config/`: Configuration file handling and validation.
  - `core_integration/`: Abstraction layer for interacting with the Greta Core Engine.
  - `output/`: Result formatting and reporting utilities.
- **Dependencies**: Relies on the Greta Core Engine for analytical processing, with loose coupling to allow independent evolution.

This structure ensures separation of concerns, where the CLI focuses on user interaction and workflow orchestration, while delegating complex analysis to the Core Engine.

## Key Components

### CLI Framework
- Built on Typer for type-safe command parsing, automatic help generation, and shell completion.
- Supports subcommands for different operations.

### Commands
- **`init`**: Initializes a new Greta project by creating a default `config.yml` file and setting up directory structure with enhanced configuration options.
- **`run`**: Executes the full analysis pipeline based on the provided configuration, processing data through ingestion, enhanced preprocessing (including automated feature engineering and advanced encoding), hypothesis search with genetic algorithms, comprehensive statistical analysis (including non-parametric tests), and enhanced narrative generation.
- **`report`**: Generates human-readable reports from analysis results, supporting various output formats including PDF reports with charts, tables, and business insights.

### Configuration Handling
- Parses and validates `config.yml` files, which specify data sources, analysis parameters, and output options.
- Provides schema validation to ensure configuration integrity before execution.

### Integration with Greta Core Engine
- Acts as a thin wrapper around the Core Engine, translating CLI inputs into Engine API calls.
- Handles data passing, parameter mapping, and result retrieval.
- Designed to be agnostic of Core Engine internals, facilitating updates and extensions.

### Output and Reporting
- Formats analysis results into structured JSON for programmatic consumption.
- Supports additional report generation for expert review, including enhanced narratives and statistical summaries.
- Generates comprehensive PDF reports with visualizations, tables, and business insights using ReportLab.

### Logging and Monitoring
- Comprehensive logging system with configurable levels (DEBUG, INFO, WARNING, ERROR).
- File-based logging with automatic log rotation and structured log entries.
- Performance timing for all major operations with detailed execution metrics.
- Error handling with informative messages and graceful degradation.

## Data Flow

The primary data flow for the `run` command is illustrated below:

```mermaid
graph TD
     A[User executes 'greta run --config config.yml'] --> B[CLI parses command and arguments]
     B --> C[Load and validate config.yml]
     C --> D[Initialize Core Engine integration]
     D --> E[Data Ingestion: Load from sources (CSV, Excel, DB)]
     E --> F[Enhanced Preprocessing: Profiling, cleaning, and automated feature engineering]
     F --> G[Hypothesis Search: Genetic algorithm with parallel execution]
     G --> H[Comprehensive Statistical Analysis: Parametric and non-parametric tests]
     H --> I[Enhanced Narrative Generation: Plain-English insights with causal analysis]
     I --> J[Collect and structure results]
     J --> K[Format output (JSON/enhanced reports/PDF)]
     K --> L[Display to stdout or save to file with logging]
```

For `init`, the flow is simpler: Parse command → Create default config → Output success message.

For `report`, the flow involves: Parse command → Load previous results → Generate formatted report → Output.

## Technology Choices

- **Programming Language**: Python 3.8+ for alignment with the Greta Core Engine and access to scientific computing libraries.
- **CLI Framework**: Typer, chosen for its modern Pythonic approach, type hint integration, and built-in features like auto-completion and help generation, ideal for professional users.
- **Configuration Format**: YAML (via PyYAML library) for human-readable, structured configuration files.
- **Output Formats**: JSON for machine-readable results; Markdown/HTML/PDF for enhanced reports with visualizations.
- **Packaging and Distribution**: setuptools for creating a pip-installable package, enabling easy distribution and installation.
- **Core Dependency**: Greta Core Engine (internal), utilizing Pandas, SciPy, DEAP, Dask, and ReportLab for advanced data processing, genetic algorithms, and report generation.
- **Logging Framework**: Python's built-in logging module with file handlers and structured logging for comprehensive monitoring.

## Design Principles

### Modularity
- Components are organized into independent modules with well-defined interfaces, allowing for isolated development, testing, and maintenance.
- Each module (e.g., config handling, output formatting) can be updated without affecting others.

### Extensibility
- Command structure uses a registry pattern, enabling easy addition of new subcommands without modifying core CLI logic.
- Integration layer employs abstraction, supporting future extensions like alternative engines or plugin systems.
- Configuration schema is designed to be forward-compatible, allowing new parameters without breaking existing setups.

### Professional Interface
- Prioritizes scriptability and headless operation for CI/CD integration and automation.
- Provides clear, actionable error messages and validation feedback.
- Emphasizes type safety and documentation to cater to expert users.

### Separation of Concerns
- CLI layer handles user interaction and workflow; Core Engine manages analytical logic.
- Output concerns are decoupled from processing, allowing flexible result presentation.

### Testability and Maintainability
- Modular design facilitates unit testing of individual components.
- Use of type hints and interfaces improves code readability and reduces bugs.
- Comprehensive logging enables debugging and monitoring of production deployments.
- Error handling with informative messages and graceful degradation ensures reliability.