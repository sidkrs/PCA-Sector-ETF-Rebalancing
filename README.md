# PCA Sector ETF Rebalancing

This project implements a dynamic sector ETF rebalancing strategy using Principal Component Analysis (PCA) and rolling derivatives. The strategy aims to outperform a benchmark index by adjusting sector weights based on the most influential factors driving market returns.

## Overview

The program analyzes a set of sector ETFs and reweights them monthly based on the highest 3-month rolling derivative of the 24-month rolling PCA loadings for three principal components (PC1, PC2, PC3). It then backtests various weight combinations to find the optimal allocation for each principal component.

## Key Features

- Uses PCA to identify the main factors driving sector returns
- Calculates rolling PCA loadings and their derivatives
- Determines optimal weights for each principal component
- Backtests the strategy against a benchmark (S&P 500)
- Provides visualizations of PCA loadings, derivatives, and performance

## Methodology

1. **Data Collection**: Gathers monthly return data for sector ETFs and the benchmark.
2. **PCA Analysis**: Performs rolling PCA on sector returns to identify principal components.
3. **Derivative Calculation**: Computes rolling derivatives of PCA loadings.
4. **Portfolio Construction**: Selects sectors with the highest derivative for each PC monthly.
5. **Weight Optimization**: Backtests different weight combinations for PCs to maximize Sharpe ratio.
6. **Performance Evaluation**: Compares the strategy's performance against the benchmark.

## Results

For the period from July 2012 (due to rolling timeframe) to January 2024:

- **Best Sharpe Ratio**: 4.94
- **Optimal PC Weights**: PC1 (10%), PC2 (84%), PC3 (6%)

## Interpretation

The high weighting of PC2 (84%) suggests that the second most influential factor in sector returns is driving the strategy's performance. This could indicate:

1. **Sector Rotation**: PC2 might capture sector rotation trends, allowing the strategy to adapt to changing market leadership.
2. **Risk Factors**: PC2 may represent important risk factors not captured by the market's primary trend (usually represented by PC1).
3. **Economic Cycles**: The second component could be sensitive to economic cycle shifts, enabling the strategy to adjust to different macroeconomic environments.

## Visualizations

1. **PCA Loadings**: Shows how each sector contributes to the principal components over time.
2. **Derivatives**: Illustrates the rate of change in PCA loadings, helping identify trending sectors.
3. **Performance vs. Benchmark**: Compares the strategy's cumulative returns against the S&P 500.

## Customization

Users can adjust:
- The derivative rolling window (default: 3 months)
- The PCA loading window (default: 24 months)
- The timeframe to backtest
- The set of sector ETFs analyzed

## Dependencies

- numpy
- pandas
- yfinance
- scikit-learn
- plotly
- statsmodels
- matplotlib

## Future Improvements

- Explore alternative derivative calculation methods
- Add more sophisticated portfolio optimization techniques
- Incorporate fundamental data to enhance sector selection
