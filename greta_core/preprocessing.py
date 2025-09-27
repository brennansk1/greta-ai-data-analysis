"""
Automated Preprocessing Module

Performs data profiling and cleaning operations automatically. Includes functions
for handling missing values, outlier detection, data type normalization, and
feature engineering. Ensures data quality and prepares datasets for hypothesis
generation and statistical testing.
"""

# Import all functions from submodules to maintain backward compatibility
from .data_profiling import *
from .missing_value_handling import *
from .outlier_detection import *
from .feature_engineering import *