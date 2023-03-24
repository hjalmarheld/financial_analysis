# Financial Analysis Project
### Erik Held & Julien Fernandez

This notebook covers our work done for the course Data Analysis in Finance at HEC Paris.

The goal of this project is to create a trading strategy based on financial and markets data. A successful strategy will have a high Sharpe ratio and low correlation to the general market. 

This project was carried out in multiple steps which will all be covered in separate sections. Firstly the provided data was cleaned to permit easier processing. Thereafter a backtesting environment was created to allow for testing of strategies and thereafter validated. A trading strategy was created and optimised on historical data before lastly being validated on an out-of-sample test set.

Testing of the final investment strategy proved that it is not a strategy for the faint of heart but one which could deliver outsized returns with a bit of luck.

### Sections

- [Data](##Data)

- [Backtesting environment](#Backtesting-environment)

- [Our Cluster Momentum Strategy](##Our-Cluster-Momentum-strategy)

- [Bayesian Optimisation](##Bayesian-optimisation)

- [Strategy Validation](##validation)

- [Conclusion](##conclusion)

## Data
The provided is data is cleaned and reformatted to facilitate further use by a cleaning function. 

This function can be found in *datacleaner.py*.

It also enabled creation of a train and test set via a *max_time* argument which exlucdes observations later than the specified date.

Below we create a training set including data up to 2017.


```python
from datacleaner import DataCreator
DataCreator(max_time='2017')
```

## Backtesting environment
A complete backtesting environment which allows for easy testing of different strategies. This can be found in *backtesting.py*. The environment is modular and works by feeding price and financial data to user-defined strategies. Starting from the beginning of the dataset, the environment gives the strategy a variable amount of historical price and financial data depending on user preference. A strategy is then required to return a series of allocations based on this data. From these series, the environment will calculate the returns of the strategy.

The adjustable parameters for testing are:
- n_prices : how many months of price data to take into account
- n_ratios : how many months of financial data to take into account
- frequency : how often we want to trade


The environment also allows for analysis of the strategy returns and can generate multiple plots and metrics for any given strategy, also comparing them to a market benchmark, which will be illustrated below.

To ensure validity of further results we want to ensure that our backtesting environment is working as intended. This is done by comparing backtest returns from a mock strategy which invests in entire market to a simple mean of the returns from the data. If the backtests results are (roughly) the same as the mean it indicates that the backtest module is working as intended.


```python
import pandas as pd
from backtest import BackTester

def entire_market(prices, ratios):
    """
    simply strategy to buy an equal amount 
    of all available stocks at any given time
    """
    all_stocks = prices['permno'].unique()
    return pd.Series({s:1/len(all_stocks) for s in all_stocks})

# initialise backtest object
backtester = BackTester()

# run backtest with entire_market strategy
backtester.rolling_test(
    strategy=entire_market,
    n_prices=1,
    n_ratios=1, 
    frequency=1
)

# analyse backtest results
metrics = backtester.analyse()
```

    100%|██████████| 323/323 [00:01<00:00, 195.44it/s]



    
![png](backtesting_files/backtesting_4_1.png)
    


The backtest of the entire market strategy performs slightly worse than the mean of the returns. 

This is probably due to some data points getting taken out from the backtest as data points are missing. However, the cumulative returns are very similar and the backtesting class seems to be working as intended.

## Our Cluster Momentum strategy

With a working backtesting system in place we created an investment strategy. The strategy combines data science and finance and consists of multiple steps which are executed as follows:

1. For any given point in time the strategy receives $n$ months of price and data and one month of financial data.
2. Financial data is standardised (0 mean, 1 variance) and passed through principal component analysis to reduce it's dimensionality, preserving $v$ variance.
3. Reduced financial data is used to classify the different stocks into $k$ different clusters.
4. Stocks are ordered by their total return over the $n$ months.
5. The stock with the highest returns from each individual cluster is placed into a portfolio and the Sharpe ratio of this portfolio over the $n$ months is calculated.
6. If it increases the Sharpe ratio, another stock from each clusted is added to the portfolio. This is done iteratively until it does not improve the Sharpe ratio.
7. Once the best amount of stocks is found, all stocks are given an equal proportion of the portfolio. For the time being, the strategy only takes long positions.

Given the structure of the strategy as well as the backtesting environment, there are a few variable parameters:

- $n$ : the number of months of price data to include, this namely affects the calculation of return and Sharpe ratios, this is passed to the backtesting environment
- $f$ : the frequency of trading, this is passed to the backtesting environment.
- $v$ : the amount of variance which is preserved when performing principal component analysis.
- $k$ : the number of clusters which are created.

Below these parameters are chosen arbitrarily in a first attempt, however, we will later try to optimise them to improve returns.


```python
from strategy import ClusterMomentum

initial_params = {
    'n' : 3, 
    'f' : 1,
    'v' : 0.9,
    'k' : 15}

# create new backtester object
backtester = BackTester()

# create strategy object
clustermomentum = ClusterMomentum(
    n_clusters=initial_params['k'],
    variance=initial_params['v']
)

# run backtest
backtester.rolling_test(
    strategy=clustermomentum.strategy,
    n_prices=initial_params['n'],
    n_ratios=1,
    frequency=initial_params['f']
)

# analyse backtest results
metrics = backtester.analyse()
```

    100%|██████████| 321/321 [00:15<00:00, 21.23it/s]



    
![png](backtesting_files/backtesting_7_1.png)
    


The first attempt of the strategy proved very volatile. While it in total delivered greater cumulative returns than the market. It does this at the cost of a significant increase in the standard deviation, which leads to it having a smaller Sharpe ratio than the general market. This is also emphasised by the immense kurtosis of the returns, which is significantly larger than that of the market. The frequent large losses are also illustrated in the drawdown curve.

## Bayesian optimisation
To improve the results of our investment strategy we want to find better parameters for the strategy and the backtest. To do this we will use Optuna, an automatic hyperparameter optimisation framework.

We will first create an objective function, in our case we will make this a backtest of the strategy for data up to 2017 and return the Sharpe ratio of this backtest. Within this objective function, we also define the variable parameters which Optuna will try to optimise using Bayesian optimisation. Optuna will then iteratively run trials and try to improve the results based on the knowlegde gained from previous trials. 

With some luck, this should give us a better Sharpe ratio than the one achieved above. 


```python
import optuna

def objective(trial):
    # the search space for hyperparameters
    v = trial.suggest_float('v', 0, 1)
    k = trial.suggest_int('k', 1, 50)
    n = trial.suggest_int('n', 2, 12)
    f = trial.suggest_int('f', 1, 12)

    # initialise strategy which suggested parameters
    clustermomentum = ClusterMomentum(
        n_clusters=k,
        variance=v
    )

    # run backtest with suggested parameters
    backtester.rolling_test(
        strategy=clustermomentum.strategy,
        n_prices=n,
        n_ratios=1,
        frequency=f,
        disable_tqdm=True
    )
    
    # return Sharpe ratio of backtest
    return backtester._get_metrics(backtester.results)['Sharpe']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, n_jobs=-1, show_progress_bar=True)
```

    [32m[I 2023-03-24 13:40:04,540][0m A new study created in memory with name: no-name-0f73e933-e4f7-4a5c-aff0-61ecd9424c74[0m
    /Users/hjalmarheld/miniconda3/envs/tf/lib/python3.10/site-packages/optuna/progress_bar.py:56: ExperimentalWarning: Progress bar is experimental (supported from v1.2.0). The interface can change in the future.
      self._init_valid()



      0%|          | 0/5 [00:00<?, ?it/s]


    [32m[I 2023-03-24 13:40:11,839][0m Trial 4 finished with value: 0.184 and parameters: {'v': 0.15414793800341697, 'k': 23, 'n': 2, 'f': 11}. Best is trial 4 with value: 0.184.[0m
    [32m[I 2023-03-24 13:40:12,946][0m Trial 2 finished with value: 0.161 and parameters: {'v': 0.3105846984703998, 'k': 28, 'n': 11, 'f': 10}. Best is trial 4 with value: 0.184.[0m
    [32m[I 2023-03-24 13:40:13,362][0m Trial 0 finished with value: 0.181 and parameters: {'v': 0.5310964487452435, 'k': 46, 'n': 6, 'f': 12}. Best is trial 4 with value: 0.184.[0m
    [32m[I 2023-03-24 13:40:14,774][0m Trial 1 finished with value: 0.195 and parameters: {'v': 0.0014147507107995505, 'k': 50, 'n': 4, 'f': 7}. Best is trial 1 with value: 0.195.[0m
    [32m[I 2023-03-24 13:40:15,223][0m Trial 3 finished with value: 0.213 and parameters: {'v': 0.43959904342292555, 'k': 17, 'n': 8, 'f': 5}. Best is trial 3 with value: 0.213.[0m



```python
old_best = {'v': 0.6573119560648449, 'k': 26, 'n': 9, 'f': 9}
```


```python
# get the best parameters from optuna
best_params = old_best

# initialise strategy with parameters
clustermomentum = ClusterMomentum(
    n_clusters=best_params['k'],
    variance=best_params['v']
)

# run backtest with suggested parameters
backtester.rolling_test(
    strategy=clustermomentum.strategy,
    n_prices=best_params['n'],
    n_ratios=1,
    frequency=best_params['f'],)

backtester.analyse()
```

    100%|██████████| 35/35 [00:01<00:00, 21.32it/s]



    
![png](backtesting_files/backtesting_12_1.png)
    


As seen above, a better parameter choice markedly improved the performance of the strategy, especially given X and Y.

## Validation

To validate the strategy improve we will compare it to our original parameters as well as the market for the entire dataset, which includes data up to the end of 2022.


```python
# create data without max time
DataCreator()
```


```python
# try with initial parameters 

# create strategy object
clustermomentum = ClusterMomentum(
    n_clusters=initial_params['k'],
    variance=initial_params['v']
)

# run backtest
backtester.rolling_test(
    strategy=clustermomentum.strategy,
    n_prices=initial_params['n'],
    n_ratios=1,
    frequency=initial_params['f']
)

# analyse backtest results
metrics = backtester.analyse()
```

    100%|██████████| 321/321 [00:12<00:00, 25.48it/s]



    
![png](backtesting_files/backtesting_16_1.png)
    


As seen in the above graphs, the initial strategy hugely outperforms the market after 2016 in terms of cumulative returns. This while decreasing the standard deviation and increasing the Sharpe ratio compared to a test on just the training set. This logically makes sense as the strategy can be seen having a large beta and that markets largely performed very well from 2017 until the Ukraine invasion and subsequent energy crisis. Further it's notable that a large share of the market returns was realised by a small amount of companies, namely the American big tech companies, it is possible that the momementum component managed to capture this increase in value and invest in these companies.

Meanwhile, the performance of the market is slightly worse compared to during just the train set.


```python
# try with best params from optuna

# initialise strategy with parameters
clustermomentum = ClusterMomentum(
    n_clusters=best_params['k'],
    variance=best_params['v']
)

# run backtest with suggested parameters
backtester.rolling_test(
    strategy=clustermomentum.strategy,
    n_prices=best_params['n'],
    n_ratios=1,
    frequency=best_params['f'],)

backtester.analyse()
```

    100%|██████████| 35/35 [00:01<00:00, 20.89it/s]



    
![png](backtesting_files/backtesting_18_1.png)
    


## Conclusion

To be done...
