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
```

## ✨ Enhanced Feature Selection Pipeline

GRETA's advanced pipeline includes all the latest upgrades:

- **🔄 Parallel Execution**: Multi-process genetic algorithm optimization
- **🧮 Dynamic Feature Engineering**: Automatic polynomial and interaction terms
- **📊 Importance Explainability**: SHAP/permutation importance rankings
- **🎯 Stability Selection**: Bootstrap validation for robust features
- **🔗 Causal Prioritization**: Causal relationship weighting
- **⚙️ Adaptive Parameters**: Dynamic GA parameter tuning
- **🔄 Multi-modal Handling**: Advanced categorical encoding

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
    "encoding_method": "target_encoding",
    "pre_filter_fraction": 0.8
  }' \
  --output enhanced_results.json
```

### Generate Different Report Formats
```bash
# PDF Report (recommended)
greta-cli report --input-file results.json --format pdf --output report.pdf

# Text Report
greta-cli report --input-file results.json --format text

# Markdown Report
greta-cli report --input-file results.json --format markdown --output report.md
```

## 🏗️ Architecture

- **greta-core**: Core analysis engine with genetic algorithms
- **greta-cli**: Command-line interface for professional workflows
- **greta-web**: Web application for interactive analysis (Streamlit)

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- deap (genetic algorithms)
- shap (explainability)
- dowhy (causal inference)
- fpdf (PDF reports)

## 🔧 Development

For developers interested in the core library:

```python
import greta_core as gc

# Direct API usage
df = gc.ingestion.load_csv('data.csv')
df_clean = gc.preprocessing.handle_missing_values(df)
hypotheses = gc.hypothesis_search.generate_hypotheses(df_clean, 'target')
```

## 📈 Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development phases and upcoming features.

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional statistical tests
- New data connectors
- Performance optimizations
- Plugin architecture