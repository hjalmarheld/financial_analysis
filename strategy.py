import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

class ClusterMomentum:
    def __init__(
        self, 
        n_clusters: int,
        variance: float):
        self.n_clusters = n_clusters
        self.variance = variance


    def _get_cluster(self, ratios):
        """
        classify the stocks into clusters based on 
        financial ratios
        """
        X = ratios.copy()
        X = X.set_index('permno')
        X = X.drop('date', axis=1)
        X = Normalizer().transform(X=X.values)
        X = PCA(n_components=self.variance).fit_transform(X)
        X = KMeans(
            n_clusters=self.n_clusters, n_init='auto', random_state=0).fit(X).labels_
        clusters = pd.Series(X, index=ratios['permno'])
        return clusters


    def _get_returns(self, prices):
        """
        get returns of individual stocks
        """
        X = prices.copy()
        X['returns'] = X['ret'] +1
        returns = X.groupby('permno')['returns'].prod()
        return returns


    def _get_n_largest(self, n, cluster_returns):
        """
        get n stocks with largest returns
        """
        return (
            cluster_returns
            .groupby('cluster')['returns']
            .nlargest(n)
            .reset_index()['permno'])


    def _sharpe_ratio(self, prices):
        """
        calculate sharpe ratio of portfolio
        """
        mean_returns = prices.mean()
        portfolio_mean = mean_returns.mean()
        cov_matrix = prices.cov()
        portfolio_std = np.sqrt(np.dot(
            np.dot(np.array([1]*len(mean_returns)), cov_matrix),
            np.array([1]*len(mean_returns)).T)) / len(mean_returns)
        return portfolio_mean / portfolio_std


    def _select_n_stocks(self, prices, cluster_returns):
        returns = prices.pivot(columns='permno', index='date', values='ret')

        # create portfolio with start with 1 position for each cluster
        # get sharpe ratio
        n_stocks = 1
        stocks = self._get_n_largest(n_stocks, cluster_returns=cluster_returns)
        sharpe = self._sharpe_ratio(returns[stocks])

        # iteratively increase number of positions
        # if it increases sharpe ratio
        while True:
            # try increasing position
            n_stocks += 1
            _stocks = self._get_n_largest(
                n=n_stocks,
                cluster_returns=cluster_returns)
            _sharpe = self._sharpe_ratio(returns[_stocks])
            if _sharpe>sharpe:
                stocks=_stocks
                sharpe=_sharpe
            else:
                break

        return stocks


    def strategy(self, prices, ratios):
        """
        complete method which will be used for backtesting
        
        get returns and clusters, create portfolio selecting
        n stocks from each cluster such that n maximises the
        sharpe ratio over the period fed to the method.
        """
        cluster_returns = pd.DataFrame({
            'cluster':self._get_cluster(ratios=ratios),
            'returns':self._get_returns(prices=prices)
            })

        long_stocks = self._select_n_stocks(
            prices=prices,
            cluster_returns=cluster_returns)

        return pd.Series({s:1/len(long_stocks) for s in long_stocks})