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

For the period analyzed:

- **Best Sharpe Ratio**: 0.872
- **Optimal PC Weights**: PC1 (4%), PC2 (95%), PC3 (1%)

Performance metrics:
- **Annualized Alpha**: 8.16% (0.0816)
- **Beta**: 0.75811
- **R-squared**: 0.43692
- **Max Drawdown**: -0.23334

## Interpretation

The high weighting of PC2 (95%) suggests that the second most influential factor in sector returns is driving the strategy's performance. This could indicate:

1. **Sector Rotation**: PC2 might capture sector rotation trends, allowing the strategy to adapt to changing market leadership.
2. **Risk Factors**: PC2 may represent important risk factors not captured by the market's primary trend (usually represented by PC1).
3. **Economic Cycles**: The second component could be sensitive to economic cycle shifts, enabling the strategy to adjust to different macroeconomic environments.

The annualized alpha of 8.16% suggests that the strategy is generating significant excess returns compared to the benchmark, after accounting for market risk. This is a strong positive indicator of the strategy's performance. The beta of 0.75811 indicates that the strategy is less volatile than the market. The R-squared value of 0.43692 suggests that about 44% of the portfolio's movements can be explained by the benchmark's movements, indicating a moderate level of correlation with the market.

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

## Data Sources

The strategy uses the following sector ETFs:
- IYW (Technology)
- IYH (Healthcare)
- IYF (Financials)
- IYC (Consumer Discretionary)
- IYZ (Communication)
- IYK (Consumer Staples)
- IYE (Energy)
- IYJ (Industrials)
- IYM (Materials)
- IDU (Utilities)
- IYR (Real Estate)

The S&P 500 index (^GSPC) is used as the benchmark.
