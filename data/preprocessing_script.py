from random import seed

import numpy as np
import pandas as pd
import scanpy as sc

from scastral.preprocessing import CountPerMilionNormalizer

seed(1234)
np.random.seed(1234)

signature = pd.read_csv('data/signature.csv')
signature = list(signature['ensembl_gene_id'])

path = 'data/old_sum/HDQP1_2'
ensemblid_key = 'gene_ids'
cell_line_name = 'HDQP1_2.csv'
prefix = ''

adata = sc.read_10x_mtx(path, prefix=prefix)

X = pd.DataFrame(adata.X.toarray(),
                 index=adata.obs.index,
                 columns=[gene.split('.')[0] for gene in adata.var[ensemblid_key]])

X = X.iloc[X.sum(axis=1) > 5000, :]
X = pd.DataFrame(CountPerMilionNormalizer().fit_transform(X),
                 index=adata.obs.index,
                 columns=[gene.split('.')[0] for gene in adata.var[ensemblid_key]])
X = X.loc[:, ~X.columns.duplicated()]
for gene in signature:
    if gene not in X.columns:
        X[gene] = 0
X.loc[:, signature].to_csv(f'data/cell_line/{cell_line_name}')
