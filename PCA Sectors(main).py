import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from itertools import product
from plotly.subplots import make_subplots
import statsmodels.api as sm
import matplotlib.pyplot as plt


def get_sector_data(start_date='2010-01-01', end_date='2024-01-31'):
    """
    Download and process sector data from Yahoo Finance.
    
    Args:
    - start_date (str): Start date for data download.
    - end_date (str): End date for data download.
    
    Returns:
    - returns (DataFrame): Monthly returns of the sectors.
    - tickers (list): List of sector tickers.
    """
    tickers = [
        "IYW",  # Technology
        "IYH",  # Healthcare
        "IYF",  # Financials
        "IYC",  # Consumer Discretionary
        "IYZ",  # Communication
        "IYK",  # Consumer Staples
        "IYE",  # Energy
        "IYJ",  # Industrials
        "IYM",  # Materials
        "IDU",  # Utilities
        "IYR"   # Real Estate
    ]
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    monthly_data = data.resample('M').last()
    returns = monthly_data.pct_change().dropna()
    return returns, tickers


def get_benchmark_data(start_date='2010-01-01', end_date='2024-01-31'):
    """
    Download and process benchmark data from Yahoo Finance.
    
    Args:
    - start_date (str): Start date for data download.
    - end_date (str): End date for data download.
    
    Returns:
    - benchmark_returns (Series): Monthly returns of the benchmark.
    """
    benchmark = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    monthly_benchmark = benchmark.resample('M').last()
    benchmark_returns = monthly_benchmark.pct_change().dropna()
    return benchmark_returns


def perform_pca(returns):
    """
    Perform PCA on the returns data.
    
    Args:
    - returns (DataFrame): Monthly returns of the sectors.
    
    Returns:
    - pca (PCA object): Fitted PCA object.
    - scaler (StandardScaler object): Fitted scaler object.
    """
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    pca = PCA(n_components=3)
    pca.fit(scaled_returns)
    return pca, scaler


def calculate_rolling_pca_loadings(returns, window=24):
    """
    Calculate rolling PCA loadings.
    
    Args:
    - returns (DataFrame): Monthly returns of the sectors.
    - window (int): Rolling window size.
    
    Returns:
    - loadings_history (DataFrame): DataFrame containing rolling PCA loadings.
    """
    loadings_history = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window:i]
        pca, _ = perform_pca(window_returns)
        loadings = pd.DataFrame(pca.components_.T, index=returns.columns, columns=['PC1', 'PC2', 'PC3'])
        loadings['Date'] = returns.index[i - 1]
        loadings_history.append(loadings)
    
    return pd.concat(loadings_history)


def plot_pca_loadings(loadings):
    """
    Plot PCA loadings over time.
    
    Args:
    - loadings (DataFrame): DataFrame containing PCA loadings.
    """
    for pc in ['PC1', 'PC2', 'PC3']:
        fig = go.Figure()
        for sector in loadings.index.unique():
            sector_loadings = loadings.loc[sector]
            fig.add_trace(go.Scatter(x=sector_loadings['Date'], y=sector_loadings[pc],
                                     mode='lines', name=sector))
        
        fig.update_layout(title=f'{pc} Loadings Over Time',
                          xaxis_title='Date',
                          yaxis_title='Loading',
                          legend_title='Sectors',
                          hovermode="x unified")
        fig.show()


def plot_cumulative_variance(pca):
    """
    Plot cumulative variance explained by principal components.
    
    Args:
    - pca (PCA object): Fitted PCA object.
    """
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_variance_ratio) + 1)),
                         y=explained_variance_ratio,
                         name='Individual explained variance'))
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio) + 1)),
                             y=cumulative_variance_ratio,
                             mode='lines+markers',
                             name='Cumulative explained variance'))
    
    fig.update_layout(title='Explained Variance by Principal Components',
                      xaxis_title='Principal components',
                      yaxis_title='Explained variance ratio',
                      barmode='overlay',
                      hovermode="x unified")
    fig.show()


def calculate_derivative(group):
    """
    Calculate the derivative of a group of values.
    
    Args:
    - group (Series): Series of values.
    
    Returns:
    - derivative (float): Calculated derivative.
    """
    return (group.iloc[-1] - group.iloc[0]) / (len(group) - 1)


def calculate_rolling_derivative(data, window):
    """
    Calculate rolling derivative for a Series using a fixed-period window.
    
    Args:
    - data (Series): Series of values.
    - window (int): Rolling window size.
    
    Returns:
    - rolling_derivative (Series): Series of rolling derivatives.
    """
    return data.rolling(window=window, min_periods=window).apply(calculate_derivative).shift(window)


def get_deriv(scores):
    """
    Calculate 3-month rolling derivative for each sector and PCA component.
    
    Args:
    - scores (DataFrame): DataFrame containing PCA scores.
    
    Returns:
    - results_reshaped (DataFrame): Reshaped DataFrame with rolling derivatives.
    """
    scores = scores.reset_index()
    pca_columns = ['PC1', 'PC2', 'PC3']
    
    results = []
    
    for sector in scores['index'].unique():
        sector_data = scores[scores['index'] == sector].set_index('Date').sort_index()
        for col in pca_columns:
            derivative = calculate_rolling_derivative(sector_data[col], window=3)  # Set rolling amount
            derivative = derivative.reset_index()
            derivative['Sector'] = sector
            derivative['PCA'] = col
            derivative = derivative.rename(columns={col: 'Value'})
            results.append(derivative)
    
    results_df = pd.concat(results, ignore_index=True)
    
    # Pivot the DataFrame to get the desired format
    results_reshaped = results_df.pivot(index=['Sector', 'Date'], columns='PCA', values='Value').reset_index()
    
    # Reorder columns to match original format
    results_reshaped = results_reshaped[['PC1', 'PC2', 'PC3', 'Date', 'Sector']]
    
    return results_reshaped


def plot_derivatives(derivatives):
    """
    Plot the derivatives for each PCA component.
    
    Args:
    - derivatives (DataFrame): DataFrame containing derivatives.
    """
    for pc in ['PC1', 'PC2', 'PC3']:
        fig = go.Figure()
        for sector in derivatives['Sector'].unique():
            sector_data = derivatives[derivatives['Sector'] == sector]
            fig.add_trace(go.Scatter(x=sector_data['Date'], y=sector_data[pc],
                                     mode='lines', name=sector))
        
        fig.update_layout(title=f'{pc} Derivatives Over Time',
                          xaxis_title='Date',
                          yaxis_title='Derivative',
                          legend_title='Sectors',
                          hovermode="x unified")
        fig.show()


def get_portfolio(derivatives):
    """
    Construct a portfolio based on the highest derivative for each PCA component.
    
    Args:
    - derivatives (DataFrame): DataFrame containing derivatives.
    
    Returns:
    - portfolio (DataFrame): DataFrame containing the portfolio.
    """
    derivatives = derivatives.dropna(subset=['PC1'])
    dates = derivatives['Date'].unique()
    
    portfolio_dict = {f'PC{i}': [] for i in range(1, 4)}
    
    for i in dates:
        date_data = derivatives[derivatives['Date'] == i]
        for j in range(1, 4):
            pc = f'PC{j}'
            max_sector = date_data.loc[date_data[pc].idxmax()]['Sector']
            portfolio_dict[pc].append(max_sector)
    
    portfolio = pd.DataFrame(portfolio_dict, index=dates)
    return portfolio


def get_returns(portfolio, returns, risk_free_rate=0.03):
    """
    Calculate the returns and Sharpe ratio for the portfolio.
    
    Args:
    - portfolio (DataFrame): DataFrame containing the portfolio.
    - returns (DataFrame): DataFrame containing the sector returns.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
    
    Returns:
    - best_portfolio_returns (Series): Series containing the best portfolio returns.
    - best_sharpe_ratio (float): Best Sharpe ratio achieved.
    - best_weights (tuple): Best weights for the portfolio.
    - best_sectors (DataFrame): DataFrame containing the best sectors.
    """
    returns = returns.loc[portfolio.index[0]:]
    dates = portfolio.index

    def calculate_sharpe_ratio(returns, risk_free_rate):
        excess_returns = returns - risk_free_rate / 12
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    best_sharpe_ratio = -np.inf
    best_portfolio_returns = None
    best_weights = None
    best_sectors = None

    weights_combinations = [w for w in product(range(101), repeat=3) if sum(w) == 100]

    for weights in weights_combinations:
        x, y, z = [w / 100 for w in weights]
        portfolio_returns = []
        sectors = []
        
        for i in range(len(dates) - 1):
            pc1_sector = portfolio.iloc[i, 0]
            pc2_sector = portfolio.iloc[i, 1]
            pc3_sector = portfolio.iloc[i, 2]

            daily_return = (
                returns.loc[dates[i + 1], pc1_sector] * x +
                returns.loc[dates[i + 1], pc2_sector] * y +
                returns.loc[dates[i + 1], pc3_sector] * z
            )
            portfolio_returns.append(daily_return)
            sectors.append([pc1_sector, pc2_sector, pc3_sector])
        
        new_dates = dates[1:]
        portfolio_returns = pd.Series(portfolio_returns, index=new_dates)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)

        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            best_portfolio_returns = portfolio_returns
            best_sectors = sectors
            best_weights = (x, y, z)

    best_sectors = pd.DataFrame(best_sectors, index=new_dates, columns=['PC1', 'PC2', 'PC3'])

    return best_portfolio_returns, best_sharpe_ratio, best_weights, best_sectors


def plot_returns(portfolio_returns, benchmark_returns):
    """
    Plot cumulative returns of the portfolio against the benchmark.
    
    Args:
    - portfolio_returns (Series): Series containing portfolio returns.
    - benchmark_returns (Series): Series containing benchmark returns.
    
    Returns:
    - combined (DataFrame): DataFrame containing both portfolio and benchmark returns.
    """
    benchmark_returns = benchmark_returns.loc[portfolio_returns.index[0]:]
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    portfolio_cumulative = (1 + portfolio_returns).cumprod()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, mode='lines', name='Portfolio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative, mode='lines', name='Benchmark'), row=1, col=1)
    fig.update_layout(title='Portfolio vs Benchmark Cumulative Returns',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      hovermode="x unified")
    fig.show()

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=portfolio_returns.index, y=np.log(portfolio_cumulative), mode='lines', name='Portfolio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_returns.index, y=np.log(benchmark_cumulative), mode='lines', name='Benchmark'), row=1, col=1)
    fig.update_layout(title='Portfolio vs Benchmark Cumulative Returns (Log Scale)',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns',
                      hovermode="x unified")
    fig.show()

    combined = pd.concat([portfolio_returns, benchmark_returns], axis=1)
    combined.columns = ['Portfolio', 'Benchmark']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined['Benchmark'], y=combined['Portfolio'], mode='markers', text=combined.index))
    fig.update_layout(title='Portfolio vs Benchmark Returns',
                      xaxis_title='Benchmark',
                      yaxis_title='Portfolio',
                      hovermode="x unified")
    fig.show()

    return combined


def get_stats(portfolio, benchmark):
    """
    Calculate performance statistics for the portfolio.
    
    Args:
    - portfolio (Series): Series containing portfolio returns.
    - benchmark (Series): Series containing benchmark returns.
    
    Returns:
    - stats (dict): Dictionary containing performance statistics.
    """
    returns = pd.concat([portfolio, benchmark], axis=1).dropna()
    returns.columns = ['Portfolio', 'Benchmark']
    
    X = sm.add_constant(returns['Benchmark'])
    model = sm.OLS(returns['Portfolio'], X).fit()
    
    stats = {
        'Alpha': model.params[0],
        'Beta': model.params[1],
        'R-squared': model.rsquared,
        'Max Drawdown': calculate_max_drawdown(returns['Portfolio'])
    }

    print(model.summary())
    plot_regression(returns, model, X)

    return stats


def calculate_max_drawdown(returns):
    """
    Calculate the maximum drawdown for a series of returns.
    
    Args:
    - returns (Series): Series of returns.
    
    Returns:
    - max_drawdown (float): Maximum drawdown.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def plot_regression(returns, model, X):
    """
    Plot the regression line for portfolio vs benchmark returns.
    
    Args:
    - returns (DataFrame): DataFrame containing returns.
    - model (OLS object): Fitted OLS model.
    - X (DataFrame): DataFrame containing independent variable with constant.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(returns['Benchmark'], returns['Portfolio'], label='Data Points', alpha=0.5)
    plt.plot(returns['Benchmark'], model.predict(X), color='red', label='Regression Line')
    plt.xlabel('Benchmark Returns')
    plt.ylabel('Portfolio Returns')
    plt.title('Portfolio vs. Benchmark Returns')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Get monthly sector data
    returns, tickers = get_sector_data()
    
    # Get monthly benchmark data
    benchmark_returns = get_benchmark_data()

    # Calculate rolling PCA loadings
    loadings = calculate_rolling_pca_loadings(returns)

    # Plot PCA loadings
    plot_pca_loadings(loadings)
    
    # Calculate and plot cumulative variance explained
    pca, _ = perform_pca(returns)
    plot_cumulative_variance(pca)

    # Print summary statistics
    #print("Summary of PCA Loadings:")
    #print(loadings.groupby(level=0).describe())

    # Print date range of the analysis
    print(f"\nDate range of analysis: {returns.index[0]} to {returns.index[-1]}")

    # Save the loadings to a CSV file
    loadings.to_csv('pca_loadings.csv')

    # Calculate derivatives
    derivatives = get_deriv(loadings)
    derivatives.to_csv('derivatives.csv')

    # Plot derivatives
    plot_derivatives(derivatives)

    # Get the portfolio
    portfolio = get_portfolio(derivatives)

    # Get the portfolio returns
    portfolio_returns, best_sharpe, best_weights, best_sectors = get_returns(portfolio, returns)

    print(f"\nBest Sharpe Ratio: {best_sharpe}")
    print(f"Best Weights: {best_weights}")

    stats = get_stats(portfolio_returns, benchmark_returns)
    print("\n", pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']))

    # Plot portfolio cumulative returns versus benchmark
    both_returns = plot_returns(portfolio_returns, benchmark_returns)

    # Save results to CSV files
    best_sectors.to_csv('best_sectors.csv')
    portfolio_returns.to_csv('best_portfolio_returns.csv')
    both_returns.to_csv('both_returns.csv')
