import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataFetcher:
    """
    Class containing all methods required to fetch
    data for a given point in time.

    Is initialised with price and ratio data
    """
    def __init__(
            self,
            price_path=config.price_output,
            ratio_path=config.ratio_output
            ):
        self.prices = pd.read_pickle(price_path)
        self.ratios = pd.read_pickle(ratio_path)
        self.dates = pd.Series(self.ratios['date'].unique()
            ).sort_values().reset_index(drop=True)
        

    def _get_dates(self, date, number):
        """
        return pertinent dates
        """
        return self.dates.loc[self.dates<date].iloc[-number:]


    def _get_entries(self, entries, dates):
        """
        return data pertinent dates
        """
        # find all matching entries
        entries = entries.loc[entries['date'].isin(dates)].copy()

        # return companies with entries for all dates
        counts = entries['permno'].value_counts() >= len(dates)
        companies_to_keep = counts.loc[counts].index
        return entries.loc[entries['permno'].isin(companies_to_keep)]


    def _match_entries(self, prices, ratios):
        """
        match data such that only entries from dates where
        all requested financial and price data is available
        """
        companies_to_keep = prices.loc[
            prices['permno'].isin(ratios['permno']), 'permno'].unique()
        prices = prices.loc[prices['permno'].isin(companies_to_keep)]
        ratios = ratios.loc[ratios['permno'].isin(companies_to_keep)]
        return prices, ratios


    def _get_data(self, date, n_prices, n_ratios):
        """
        wrap all data fetching for specific date
        """
        price_dates = self._get_dates(
            date=date,
            number=n_prices)

        _prices = self._get_entries(
            entries=self.prices,
            dates=price_dates)

        ratio_dates = self._get_dates(
            date=date,
            number=n_ratios)

        _ratios = self._get_entries(
            entries=self.ratios,
            dates=ratio_dates)

        _prices, _ratios = self._match_entries(
            prices=_prices,
            ratios=_ratios)

        return _prices, _ratios


class Analyser:
    """
    Class with methods to analyse backtesting results
    """
    def _rolling_sharpe(self, returns, window=12):
        """
        get rolling sharpe over 12 months
        """
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        sharpe_ratio = np.sqrt(window) * (rolling_mean / rolling_std)
        return sharpe_ratio


    def _cumulative_returns(self, returns):
        """
        get cumulative returns
        """
        return (returns+1).cumprod()


    def _plot_drawdown_curve(self, returns, market, ax):
        """
        plot drawdown curve for a series of returns.
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        cumulative_market = (1 + market).cumprod()

        running_max = pd.Series(cumulative_returns).cummax()
        running_max_market = pd.Series(cumulative_market).cummax()


        drawdown = (cumulative_returns - running_max) / running_max
        drawdown_market = (cumulative_market - running_max_market) / running_max_market

        drawdown = drawdown.dropna()
        drawdown_market = drawdown_market.dropna()

        ax.plot(drawdown_market)
        ax.plot(drawdown)
        #ax.fill_between(drawdown.index, drawdown, 0, alpha=0.1)
        ax.set(title='Drawdown Curve')


    def _get_metrics(self, returns):
        """
        get some standard return metrics
        """
        metrics = pd.Series({
            'Mean returns' : returns.mean(),
            'Standard deviation' : returns.std(),
            'Sharpe' : returns.mean() / returns.std() * 12**.5,
            'Skew' : returns.skew(),
            'Kurtosis' : returns.kurtosis()
        })

        return metrics.round(3)


    def _plot_metrics(self, metrics, ax):
        """
        plot table of return metrics
        """
        ax.axis('off')
        ax.axis('tight')

        #create table
        ax.table(
            rowLabels=metrics.index,
            cellText=metrics.values,
            cellLoc='center',
            colLabels=metrics.columns,
            bbox=[0.2,0,0.8,1])
    
        
    def plot_results(self, returns, benchmark, investments):
        """
        method drawing all of the plots for backtest
        analytics, should be called from backtest class
        """
        market = benchmark.mean(axis=1)

        # create 2x3 plotgrid
        fig, axs = plt.subplot_mosaic(
            mosaic="""
            AB
            CD
            EE
            """,
            figsize=(8,12))
        fig.set_dpi(120)

        # cumulative returns plot
        benchmark_cumulative = self._cumulative_returns(market)
        strategy_cumulative = self._cumulative_returns(returns)
        benchmark_cumulative.plot(ax=axs['A'], label='Market')
        strategy_cumulative.plot(ax=axs['A'], label='Backtest')
        axs['A'].legend()
        axs['A'].set_title('Cumulative Returns')
        axs['A'].set_xlabel('')

        # rolling sharpe plot
        benchmark_cumulative = self._rolling_sharpe(market)
        strategy_cumulative = self._rolling_sharpe(returns)
        benchmark_cumulative.plot(ax=axs['B'], label='Market')
        strategy_cumulative.plot(ax=axs['B'], label='Backtest')
        axs['B'].set_title('Rolling Sharpe')
        axs['B'].set_xlabel('')

        # drawdown plot
        self._plot_drawdown_curve(returns, market, ax=axs['C'])

        # number of stocks plot
        investments['allocs'].apply(len).plot(ax=axs['D'])
        axs['D'].set_title('Number of investments')

        # metrics plot 
        benchmark_metrics = self._get_metrics(market)
        strategy_metrics = self._get_metrics(returns)

        strategy_metrics['Beta'] = round(
            market.loc[returns.index].cov(returns) / market.var(), 3)
        benchmark_metrics['Beta'] = '-'

        metrics = pd.DataFrame(
            {'Strategy':strategy_metrics, 'Market':benchmark_metrics})
        self._plot_metrics(metrics=metrics, ax=axs['E'])

        # title and layout
        fig.suptitle('Backtest Results', fontsize=24)
        fig.tight_layout()

        plt.show()


class BackTester(DataFetcher, Analyser):
    def __init__(self):
        DataFetcher.__init__(self)
        self.returns = self.prices.pivot(
            index='date',
            values='ret',
            columns='permno')


    def _results(self):
        """
        calculate results given allocations
        """
        portfolio_returns = {}
        for _, data in self.investments.iterrows():
            # get pertinent return data
            _returns = self.returns.loc[
                (data['buy_date']<self.returns.index) & (self.returns.index<=data['sell_date']),
                data['allocs'].index]
            # multiply by allocations, get cumulative
            _returns = _returns.multiply(data['allocs'])
            _returns = _returns.sum(axis=1)
            for date in _returns.index:
                portfolio_returns[date]=_returns.loc[date]
            #portfolio_returns[data['sell_date']]=_returns
        # return series of returns over all periods
        return pd.Series(portfolio_returns)


    def rolling_test(
            self,
            strategy,
            n_prices=1,
            n_ratios=1,
            frequency=1,
            disable_tqdm=False
            ):
        """
        main backtesting method, fetches data at
        desired frequency, investment strategy and
        returns allocations given strategy
        """
        investments = pd.Series(dtype='object')
        # jump ahead in dates to allow lookback
        _dates = self.dates.iloc[max(n_prices, n_ratios):]
        # loop over dates
        for _date in tqdm(_dates[::frequency], disable=disable_tqdm):
            # get data for date
            _prices, _ratios = self._get_data(
                date=_date,
                n_prices=n_prices,
                n_ratios=n_ratios)
            # get investments for date
            investments.loc[_date] = strategy(
                prices=_prices,
                ratios=_ratios)
            self._test_prices = _prices
            self._test_ratios = _ratios
        # cleaning data, specifying dates etc
        investments = pd.DataFrame(investments, columns=['allocs'])
        investments['buy_date'] = investments.index
        investments['sell_date'] = investments['buy_date'].shift(-1)
        investments = investments.dropna()
        self.investments = investments
        self.results = self._results()


    def analyse(self):
        """
        analyse return using the Analyser class
        """
        try:
            _ = self.results
        except NameError:
            print('Unable to analyse results, has backtest been run?')

        self.plot_results(
            returns=self.results,
            benchmark=self.returns,
            investments=self.investments)