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
    benchmark = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    monthly_benchmark = benchmark.resample('M').last()
    benchmark_returns = monthly_benchmark.pct_change().dropna()
    return benchmark_returns


def perform_pca(returns):
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    pca = PCA(n_components=3)
    pca.fit(scaled_returns)
    return pca, scaler


def calculate_rolling_pca_loadings(returns, window=24):
    loadings_history = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window:i]
        pca, _ = perform_pca(window_returns)
        loadings = pd.DataFrame(pca.components_.T, index=returns.columns, columns=['PC1', 'PC2', 'PC3'])
        loadings['Date'] = returns.index[i - 1]
        loadings['Sector'] = loadings.index  # Add the sector names as a column
        loadings_history.append(loadings)
    
    return pd.concat(loadings_history).reset_index(drop=True)

def plot_pca_loadings(loadings):
    for pc in ['PC1', 'PC2', 'PC3']:
        fig = go.Figure()
        for sector in loadings['Sector'].unique():  # Use 'Sector' instead of 'index'
            sector_loadings = loadings[loadings['Sector'] == sector]
            fig.add_trace(go.Scatter(x=sector_loadings['Date'], y=sector_loadings[pc],
                                     mode='lines', name=sector))
        
        fig.update_layout(title=f'{pc} Loadings Over Time',
                          xaxis_title='Date',
                          yaxis_title='Loading',
                          legend_title='Sectors',
                          hovermode="x unified")
        fig.show()


def plot_cumulative_variance(pca):
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
    return (group.iloc[-1] - group.iloc[0]) / (len(group) - 1)


def calculate_rolling_derivative(data, window):
    # Applying the derivative calculation with a shift as in the second script
    return data.rolling(window=window, min_periods=window).apply(calculate_derivative).shift(window)


def get_deriv(scores):
    scores = scores.reset_index(drop=True)
    pca_columns = ['PC1', 'PC2', 'PC3']
    
    results = []
    
    for sector in scores['Sector'].unique():  # Corrected from 'index' to 'Sector'
        sector_data = scores[scores['Sector'] == sector].set_index('Date').sort_index()
        for col in pca_columns:
            derivative = calculate_rolling_derivative(sector_data[col], window=3)  # Set rolling amount
            derivative = derivative.reset_index()
            derivative['Sector'] = sector
            derivative['PCA'] = col
            derivative = derivative.rename(columns={col: 'Value'})
            results.append(derivative)
    
    results_df = pd.concat(results, ignore_index=True)
    
    results_reshaped = results_df.pivot(index=['Sector', 'Date'], columns='PCA', values='Value').reset_index()
    
    results_reshaped = results_reshaped[['PC1', 'PC2', 'PC3', 'Date', 'Sector']]
    
    return results_reshaped


def plot_derivatives(derivatives):
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
    returns = returns.loc[portfolio.index[0]:]
    dates = portfolio.index

    def calculate_sharpe_ratio(returns, risk_free_rate):
        excess_returns = returns - risk_free_rate / 12  # Monthly data adjustment
        return np.sqrt(12) * excess_returns.mean() / excess_returns.std()

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
            try:
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
            except KeyError:
                continue
        
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
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def plot_regression(returns, model, X):
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
    returns, tickers = get_sector_data()
    benchmark_returns = get_benchmark_data()
    loadings = calculate_rolling_pca_loadings(returns)
    plot_pca_loadings(loadings)
    pca, _ = perform_pca(returns)
    plot_cumulative_variance(pca)
    print(f"\nDate range of analysis: {returns.index[0]} to {returns.index[-1]}")
    loadings.to_csv('pca_loadings.csv')
    derivatives = get_deriv(loadings)
    derivatives.to_csv('derivatives.csv')
    plot_derivatives(derivatives)
    portfolio = get_portfolio(derivatives)
    portfolio_returns, best_sharpe, best_weights, best_sectors = get_returns(portfolio, returns)
    print(f"\nBest Sharpe Ratio: {best_sharpe}")
    print(f"Best Weights: {best_weights}")
    stats = get_stats(portfolio_returns, benchmark_returns)
    print("\n", pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']))
    both_returns = plot_returns(portfolio_returns, benchmark_returns)
    best_sectors.to_csv('best_sectors.csv')
    portfolio_returns.to_csv('best_portfolio_returns.csv')
    both_returns.to_csv('both_returns.csv')
