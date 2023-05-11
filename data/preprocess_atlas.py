from random import seed

import numpy as np
import pandas as pd
from tqdm import tqdm

from scastral.preprocessing import CountPerMilionNormalizer
from scastral.utils import filter_data
from scastral.utils import load_adata

seed(1234)
np.random.seed(1234)

"""
preprocess the brca atlas obtained from figshare

"""


adata = load_adata('data', mat_file='matrix.mtx.gz', barc_file='barcodes.tsv.gz', feat_file='features.tsv.gz')
signature = pd.read_csv('data/signature.csv')
signature = list(signature['ensembl_gene_id'])

adata.obs['cell_line'] = [barcode.split('_')[0] for barcode in adata.obs[0]]
adata.obs['barcode'] = [barcode.split('_')[1] for barcode in adata.obs[0]]

adatas = {
    cell_line: adata[adata.obs['cell_line'] == cell_line, :]
    for cell_line in adata.obs['cell_line'].unique()
}
del adata

for cell_line in tqdm(adatas):
    cell_filter, _ = filter_data(adatas[cell_line].X)
    curr = adatas[cell_line][cell_filter, :]
    X = CountPerMilionNormalizer().fit_transform(curr.X)
    X = pd.DataFrame(X, columns=curr.var[0], index=curr.obs['barcode'])
    for gene in signature:
        if gene not in X.columns:
            X[gene] = 0
    X = X.loc[:, signature]  # subset to signature
    X.to_csv(f'data/cell_line/{cell_line}.csv')
