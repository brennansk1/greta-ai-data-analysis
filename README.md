# GRETA - Automated Data Analysis with Genetic Algorithms

GRETA is an AI-powered data analysis tool that acts as your automated data strategist. Using advanced genetic algorithms and statistical methods, GRETA discovers meaningful patterns and relationships in your data, providing actionable insights with plain-English explanations.

## 🚀 Quick Start with CLI

The easiest way to use GRETA is through the command-line interface:

### 1. Install
```bash
pip install -e .
```

### 2. Initialize Project
```bash
# Create a config file for your dataset
greta-cli init --data-source your_data.csv --target-column target_variable --output config.yml
```

### 3. Run Analysis
```bash
# Execute the enhanced feature selection pipeline
greta-cli run --config config.yml --output results.json
```

### 4. Generate Report
```bash
# Create a human-readable report
greta-cli report --input-file results.json --format pdf --output analysis_report.pdf

# Generate different report formats
greta-cli report --input-file results.json --format pdf --output report.pdf
greta-cli report --input-file results.json --format text
greta-cli report --input-file results.json --format markdown --output report.md
```

## ✨ Enhanced Feature Selection Pipeline

GRETA's advanced pipeline includes all the latest upgrades:

- **🔄 Parallel Execution**: Multi-process genetic algorithm optimization
- **🧮 Dynamic Feature Engineering**: Automatic polynomial, trigonometric, logarithmic, and interaction terms
- **📊 Importance Explainability**: SHAP/permutation importance rankings with consensus analysis
- **🎯 Stability Selection**: Bootstrap validation for robust features
- **🔗 Causal Prioritization**: Causal relationship weighting with DoWhy integration
- **⚙️ Adaptive Parameters**: Dynamic GA parameter tuning
- **🔄 Multi-modal Handling**: Advanced categorical encoding (one-hot, label, ordinal, frequency, target-mean)
- **📈 Non-parametric Tests**: Mann-Whitney U, Kruskal-Wallis, and permutation tests for robust statistical validation
- **📋 Automated Feature Generation**: Intelligent feature creation with Dask support for large datasets

## 📊 Real-World Validation

Successfully tested on:
- **Telco Customer Churn**: 7,043 customers, identified key churn predictors
- **Online Retail Transactions**: 500K+ transactions, discovered pricing insights

## 🛠️ Advanced CLI Usage

### Enable All Enhanced Features
```bash
greta-cli run --config config.yml \
  --override '{
    "hypothesis_search": {"pop_size": 50, "num_generations": 20, "n_processes": 2},
    "bootstrap_iterations": 3,
    "use_causal_prioritization": true,
    "adaptive_params": true,
    "encoding_method": "target_mean",
    "numeric_transforms": ["polynomial", "trigonometric", "logarithmic"],
    "max_interactions": 5,
    "use_nonparametric_tests": true
  }' \
  --output enhanced_results.json
```

### Generate Different Report Formats
```bash
# PDF Report (recommended) - includes charts, tables, and business insights
greta-cli report --input-file results.json --format pdf --output report.pdf

# Text Report - plain text summary
greta-cli report --input-file results.json --format text

# Markdown Report - formatted markdown with enhanced narratives
greta-cli report --input-file results.json --format markdown --output report.md
```

## 🏗️ Architecture

- **greta-core**: Core analysis engine with genetic algorithms
- **greta-cli**: Command-line interface for professional workflows

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- deap (genetic algorithms)
- shap (explainability)
- dowhy (causal inference)
- fpdf (PDF reports)
- reportlab (enhanced PDF generation)
- dask (large dataset support)
- plotly (interactive visualizations)

## 🔧 Development

For developers interested in the core library:

```python
import greta_core as gc

# Direct API usage with enhanced features
df = gc.ingestion.load_csv('data.csv')
df_clean = gc.preprocessing.handle_missing_values(df)

# Enhanced feature engineering
df_engineered, metadata = gc.preprocessing.prepare_features_for_modeling(
    df_clean, target=df_clean['target'],
    encoding_method='target_mean',
    numeric_transforms=['polynomial', 'trigonometric']
)

# Generate hypotheses with non-parametric validation
hypotheses = gc.hypothesis_search.generate_hypotheses(df_engineered, 'target')

# Generate enhanced narratives
narratives = gc.narratives.generate_summary_narrative(hypotheses, df_engineered.columns.tolist())
```

## 📈 Current Status

GRETA is currently in **Phase 3: Advanced Analytics & UX** with the enhanced feature selection pipeline fully implemented and validated. The project is transitioning to **Phase 4: Enterprise Readiness & Scalability**, focusing on big data support, job queue systems, and collaborative features.

### Recent Achievements (2025-2026)
- ✅ **Enhanced Feature Selection Pipeline**: Parallel GA optimization, dynamic feature engineering, SHAP importance, bootstrap validation, causal prioritization
- ✅ **Advanced Statistical Methods**: Non-parametric tests (Mann-Whitney U, Kruskal-Wallis, permutation tests)
- ✅ **Comprehensive PDF Reporting**: Business insights with charts and automated generation
- ✅ **AutoML Framework**: Complete AutoML pipeline with model registry, hyperparameter tuning, and evaluation (implementation complete, integration pending)

### Next Priorities
- 🔄 **AutoML Integration**: CLI and web app integration for automated machine learning
- 🔄 **Scalable Backend**: Dask/Spark support for large datasets
- 🔄 **Job Queue System**: Asynchronous processing with Celery/Redis
- 🔄 **Enterprise Features**: Team collaboration, security, and performance optimization

See [ROADMAP.md](ROADMAP.md) for detailed development phases and upcoming features.

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional statistical tests (parametric and non-parametric)
- Enhanced feature engineering methods
- New data connectors and ingestion formats
- Performance optimizations for large datasets
- Plugin architecture and extensibility
- Advanced causal inference methods
- Improved narrative generation and explainability