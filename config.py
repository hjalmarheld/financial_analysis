from pathlib import Path

price_input = Path('data', 'crsp_top1000.csv')
price_output = Path('data', 'cleaned_prices.pkl')

ratio_input = Path('data', 'financialratios.csv')
ratio_output = Path('data', 'cleaned_ratios.pkl')