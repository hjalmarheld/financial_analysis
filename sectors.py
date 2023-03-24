txt = """A. Agriculture, Forestry, & Fishing
01 Agricultural Production – Crops
02 Agricultural Production – Livestock
07 Agricultural Services
08 Forestry
09 Fishing, Hunting, & Trapping

B. Mining
10 Metal, Mining
12 Coal Mining
13 Oil & Gas Extraction
14 Nonmetallic Minerals, Except Fuels

C. Construction
15 General Building Contractors
16 Heavy Construction, Except Building
17 Special Trade Contractors

D. Manufacturing
20 Food & Kindred Products
21 Tobacco Products
22 Textile Mill Products
23 Apparel & Other Textile Products
24 Lumber & Wood Products
25 Furniture & Fixtures
26 Paper & Allied Products
27 Printing & Publishing
28 Chemical & Allied Products
29 Petroleum & Coal Products
30 Rubber & Miscellaneous Plastics Products
31 Leather & Leather Products
32 Stone, Clay, & Glass Products
33 Primary Metal Industries
34 Fabricated Metal Products
35 Industrial Machinery & Equipment
36 Electronic & Other Electric Equipment
37 Transportation Equipment
38 Instruments & Related Products
39 Miscellaneous Manufacturing Industries

E. Transportation & Public Utilities
40 Railroad Transportation
41 Local & Interurban Passenger Transit
42 Trucking & Warehousing
43 U.S. Postal Service
44 Water Transportation
45 Transportation by Air
46 Pipelines, Except Natural Gas
47 Transportation Services
48 Communications
49 Electric, Gas, & Sanitary Services

F. Wholesale Trade
50 Wholesale Trade – Durable Goods
51 Wholesale Trade – Nondurable Goods
52 Building Materials & Gardening Supplies
53 General Merchandise Stores
54 Food Stores
55 Automative Dealers & Service Stations
56 Apparel & Accessory Stores
57 Furniture & Homefurnishings Stores
58 Eating & Drinking Places
59 Miscellaneous Retail

H. Finance, Insurance, & Real Estate
60 Depository Institutions
61 Nondepository Institutions
62 Security & Commodity Brokers
63 Insurance Carriers
64 Insurance Agents, Brokers, & Service
65 Real Estate
67 Holding & Other Investment Offices

I. Services
70 Hotels & Other Lodging Places
72 Personal Services
73 Business Services
75 Auto Repair, Services, & Parking
76 Miscellaneous Repair Services
78 Motion Pictures
79 Amusement & Recreation Services
80 Health Services
81 Legal Services
82 Educational Services
83 Social Services
84 Museums, Botanical, Zoological Gardens
86 Membership Organizations
87 Engineering & Management Services
88 Private Households
89 Services, Not Elsewhere Classified

J. Public Administration
91 Executive, Legislative, & General
92 Justice, Public Order, & Safety
93 Finance, Taxation, & Monetary Policy
94 Administration of Human Resources
95 Environmental Quality & Housing
96 Administration of Economic Programs
97 National Security & International Affairs
98 Zoological Gardens

K. Nonclassifiable Establishments
99 Non-Classifiable Establishments"""

cats=[]
nums=[]
for l in txt.splitlines():
    if l=='':
        pass
    elif l[0] not in '1234567890':
        cat=l
    else:
        cats.append(cat)
        nums.append(int(l[:2]))

sectors = pd.DataFrame({'Sector':cats,'sic2':nums})


sectors = sectors.set_index('sic2')

sectors = (
    backtester.ratios[['permno','sic2']]
    .merge(sectors, left_on='sic2',right_index=True)
    .drop('sic2', axis=1)
    .drop_duplicates()
    .set_index('permno')
)

def calculate_sectors(investments, sectors):

    _investments = []
    for date in investments.index:
        _sectors = (
            pd.DataFrame(investments.loc[date, 'allocs'])
            .merge(sectors, left_index=True, right_index=True)
        )

        _investments.append(pd.DataFrame(_sectors['Sector'].value_counts(normalize=True)).T)

    sector_investments = pd.concat(_investments)
    sector_investments.index = investments.index
    return sector_investments