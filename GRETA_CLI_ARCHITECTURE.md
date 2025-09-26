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
- **`init`**: Initializes a new Greta project by creating a default `config.yml` file and setting up directory structure.
- **`run`**: Executes the full analysis pipeline based on the provided configuration, processing data through ingestion, preprocessing, hypothesis search, statistical analysis, and narrative generation.
- **`report`**: Generates human-readable reports from analysis results, supporting various output formats (e.g., Markdown, JSON summaries).

### Configuration Handling
- Parses and validates `config.yml` files, which specify data sources, analysis parameters, and output options.
- Provides schema validation to ensure configuration integrity before execution.

### Integration with Greta Core Engine
- Acts as a thin wrapper around the Core Engine, translating CLI inputs into Engine API calls.
- Handles data passing, parameter mapping, and result retrieval.
- Designed to be agnostic of Core Engine internals, facilitating updates and extensions.

### Output and Reporting
- Formats analysis results into structured JSON for programmatic consumption.
- Supports additional report generation for expert review, including narratives and statistical summaries.

## Data Flow

The primary data flow for the `run` command is illustrated below:

```mermaid
graph TD
    A[User executes 'greta run --config config.yml'] --> B[CLI parses command and arguments]
    B --> C[Load and validate config.yml]
    C --> D[Initialize Core Engine integration]
    D --> E[Data Ingestion: Load from sources (CSV, Excel, DB)]
    E --> F[Preprocessing: Profiling and cleaning]
    F --> G[Hypothesis Search: Genetic algorithm execution]
    G --> H[Statistical Analysis: Tests on hypotheses]
    H --> I[Narrative Generation: Plain-English insights]
    I --> J[Collect and structure results]
    J --> K[Format output (JSON/report)]
    K --> L[Display to stdout or save to file]
```

For `init`, the flow is simpler: Parse command → Create default config → Output success message.

For `report`, the flow involves: Parse command → Load previous results → Generate formatted report → Output.

## Technology Choices

- **Programming Language**: Python 3.8+ for alignment with the Greta Core Engine and access to scientific computing libraries.
- **CLI Framework**: Typer, chosen for its modern Pythonic approach, type hint integration, and built-in features like auto-completion and help generation, ideal for professional users.
- **Configuration Format**: YAML (via PyYAML library) for human-readable, structured configuration files.
- **Output Formats**: JSON for machine-readable results; Markdown/HTML for reports.
- **Packaging and Distribution**: setuptools for creating a pip-installable package, enabling easy distribution and installation.
- **Core Dependency**: Greta Core Engine (internal), utilizing Pandas, SciPy, and DEAP for data processing and genetic algorithms.

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