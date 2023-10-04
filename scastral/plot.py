import os
import pickle as pkl
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
seed(1234)
np.random.seed(1234)
torch.random.manual_seed(1234)

import dill
with open('models/scASTRAL_final_pipeline.pkl', 'rb') as handle:
    serialized = handle.read()
model = dill.loads(serialized)

#model = pkl.load(open('models/scASTRAL_final_pipeline.pkl','rb'))

response = pd.read_csv('data/afatinib.csv')
to_test = ["HCC1937","HCC38","HCC1187","CAL51","DU4475","HS578T",
           "BT549","MDAMB436", "BT20","HDQP1","CAL851","HCC1143"]

#"SUM159PT","SUM229PE" ,"SUM1315MO2","SUM185","SUM149PT","HCC70",

signature = pd.read_csv('data/signature_374.csv')
signature = list(signature['ensembl_gene_id'])

response['cl'] = [p.replace('-','') for p in response['Cell line']]
response['log(IC50)'] = np.log2(response['IC50'])
response = response.loc[ [p in to_test for p in response['cl']] ,['cl','log(IC50)'] ]
response['score']=np.nan

tot = 0
for file in os.listdir('data/rawdata'):
    cell_line,extension = file.split('.')
    if extension == 'csv' and cell_line in to_test:
        df = pd.read_csv(f"data/rawdata/{file}",index_col=0)
        df = df.loc[:,signature]
        tot += df.shape[0]
        response.loc[response['cl']==cell_line,'score']= model.predict_log_proba(df)[:,1].mean()

response = response.dropna()
test = pearsonr(response['log(IC50)'],response['score'])
fig, ax = plt.subplots()
response.plot(kind='scatter',y='log(IC50)',x='score', title=f" SCC: {test.statistic:.4f} - pvalue: {test.pvalue:.4f}",ax=ax)
for idx, row in response.iterrows():
    ax.annotate(row['cl'], (row['log(IC50)'], row['score']),xytext=(5,-5),
                textcoords='offset points', family='sans-serif', fontsize=12)
plt.show()