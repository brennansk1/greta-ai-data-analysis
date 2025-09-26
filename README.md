# Greta Core Engine

Automated Data Analysis Library with Genetic Algorithms

## Installation

```bash
pip install -e .
```

## Dependencies

- pandas
- numpy
- scipy
- scikit-learn
- deap
- openpyxl

## Usage

```python
import greta_core as gc

# Load data
df = gc.ingestion.load_csv('data.csv')

# Preprocess
df_clean = gc.preprocessing.handle_missing_values(df)

# Generate hypotheses
# ... (full pipeline)
```

## Modules

- `ingestion`: Data loading from CSV, Excel
- `preprocessing`: Data profiling and cleaning
- `hypothesis_search`: Genetic algorithm for hypothesis generation
- `statistical_analysis`: Statistical tests (t-tests, ANOVA)
- `narrative_generation`: Plain-English insights