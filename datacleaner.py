import config
import pandas as pd

# ratio clean up

def DataCreator(max_time='2023'):
    ratios = pd.read_csv(config.ratio_input, sep='	')

    # convert dates to pandas datetime, only keeping date of publication
    ratios['date'] = (
        pd.to_datetime(ratios['public_date'], format='%d%b%Y')
        + pd.tseries.offsets.MonthEnd(0))
    ratios = ratios.drop(['adate', 'qdate', 'public_date'], axis=1)

    # remove columns with more than max_na missing values
    max_na = 20_000
    ratios = ratios.dropna(axis=1, thresh=len(ratios)-max_na)

    # remove rows still missing values
    ratios = ratios.dropna()


    # price clean up

    prices = pd.read_csv(
        config.price_input,
        sep='	',
        low_memory=False)

    # convert dates to pandas datetime remove other formats,
    # settings dates to last of month to match ratios
    prices['date'] = (
        pd.to_datetime(prices['date'], format='%d%b%Y')
        + pd.tseries.offsets.MonthEnd(0)
    )
    prices = prices.drop(['year', 'month'], axis=1)

    # convert returns to numpy float
    prices['ret'] = (
        pd.to_numeric(
            # fix decimal strings
            prices['ret'].str.replace('.','0.', regex=False),
            # force errors to nans
            errors='coerce'
        # drop nans
        ).dropna()
    )


    # final clean up and save

    # move sic2 to ratios
    sectors = prices[['permno', 'sic2']].groupby('permno')['sic2'].first()
    prices = prices.drop('sic2', axis=1)
    ratios = ratios.merge(sectors, left_on='permno', right_index=True, how='inner')

    # set max time to create train set if needed
    prices = prices.loc[prices['date']<=max_time]
    ratios = ratios.loc[ratios['date']<=max_time]

    # save to pickles
    prices.to_pickle(config.price_output)
    ratios.to_pickle(config.ratio_output)