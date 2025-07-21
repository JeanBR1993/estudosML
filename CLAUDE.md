# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational machine learning project focused on implementing regression algorithms from scratch. The project uses Python with Jupyter notebooks and explores the California Housing dataset for practical applications.

## Dependencies

The project uses the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

To install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Project Structure

- `estudoML.ipynb` - Main notebook implementing linear regression from scratch with gradient descent
- `teste.ipynb` - Performance testing notebook comparing vectorized vs loop-based operations
- `explicacao_ml_detalhada.md` - Theoretical foundations of ML concepts
- `exemplos-praticos-ml.md` - Practical implementation guide

## Development Commands

### Running Notebooks
```bash
jupyter notebook
```

### Working with Virtual Environment
```bash
# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
```

## Code Architecture

### Key Implementation Details

1. **Gradient Descent Implementation** (`estudoML.ipynb`):
   - Manual implementation without libraries for educational purposes
   - Supports multiple hypotheses with different feature combinations
   - Implements both MinMax and Z-score normalization
   - Uses vectorized operations for performance

2. **Feature Engineering**:
   - K-means clustering for geographical features (latitude/longitude)
   - Correlation analysis for feature selection
   - Normalization strategies for different scales

3. **Performance Considerations**:
   - Always prefer vectorized NumPy operations over loops
   - The project demonstrates ~17x speedup with vectorization

### Current Hypotheses Being Tested

The project tests multiple regression models with increasing complexity:
- Hypothesis 1: Only median income
- Hypothesis 2: Median income + housing age + total rooms
- Hypothesis 3: All features
- Hypothesis 4: All features + geographical clustering

## Important Notes

- This is a learning project - implementations prioritize understanding over production readiness
- Manual gradient descent implementations are for educational purposes
- Always validate results against scikit-learn implementations
- Focus on understanding the mathematical foundations before optimizing code

## ALWAYS THINK DEEPLY TO ANSWER THE USER