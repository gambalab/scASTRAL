import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

valid_set = [
    'SUM149PT',
    'HCC38',
    'HCC70',
    'CAL851',
    'SUM229PE',
    'HDQP1',
    'BT20',
    'HCC1937',
    'SUM185PE',
    'HCC1143',
    'BT549',
    'SUM159PT',
    'CAL51',
    'SUM1315MO2',
    'HCC1187',
    'MDAMB436',

]

signature = pd.read_csv('data/signature_374.csv')
signature = list(signature['ensembl_gene_id'])

response = pd.read_csv('data/validation_set/afatinib.csv')
response['cl'] = [p.replace('-', '') for p in response['Cell line']]
response['log(IC50)'] = np.log2(response['IC50'])
response = response.loc[:, ['cl', 'log(IC50)']]
response['score'] = np.nan

path = 'data/validation_set/preprocessed'


model = pkl.load(open(f"models/scastral.pkl", 'rb'))

for cell_line in valid_set:
    df = pd.read_csv(f'{path}/{cell_line}.csv',index_col=0)
    df = df[signature].to_numpy()
    proba = model.predict_proba(df)[:, 1]
    response.loc[response['cl'] == cell_line, 'score'] = np.mean(proba > .75)

response = response.dropna()
test = pearsonr(response['log(IC50)'], response['score'])
response.plot.scatter(y='log(IC50)', x='score', title=f" spearman: {test.statistic:.4f} - p: {test.pvalue:.4f}")
rss = linregress(response['score'], response['log(IC50)'])
x = np.array([response['score'].min(), response['score'].max()])
plt.plot(x, rss.intercept + rss.slope * x, 'r')
for idx, row in response.iterrows():
    plt.text(row['score'], row['log(IC50)'] - .1, row['cl'], va='top', ha='center', fontsize=8)
plt.savefig('correlation.pdf')
plt.show()
