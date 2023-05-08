
import pandas as pd
from random import seed
import numpy as np
import torch
from scastral.utils import filter_data
from scastral.utils import load_adata
from scastral.preprocessing import CountPerMilionNormalizer
import pickle as pkl
seed(1234)
np.random.seed(1234)
torch.random.manual_seed(1234)


model = pkl.load(open('models/scASTRAL_pipeline.sk','rb'))
adata = load_adata('data', mat_file='matrix.mtx.gz', barc_file='barcodes.tsv.gz', feat_file='features.tsv.gz')
signature = pd.read_csv('data/signature.csv')
signature = list(signature['ensembl_gene_id'])



adata.obs['cell_line'] = [ barcode.split('_')[0] for barcode in adata.obs[0]]
adata.obs['barcode']=[ barcode.split('_')[1] for barcode in adata.obs[0]]


adatas = {
    cell_line:adata[adata.obs['cell_line']==cell_line,:]
    for cell_line in adata.obs['cell_line'].unique()
}
del adata

del adata
#%%
for cell_line in adatas:
    cell_filter,_ = filter_data(adatas[cell_line].X)
    X = adatas[cell_line][cell_filter,:]
    X = CountPerMilionNormalizer().fit_transform(X)
    X = pd.DataFrame(X,columns=adatas[cell_line].var[0],index=adatas[cell_line].obs['barcode'])
    X = X.loc[:,signature] # subset to signature
    X.to_csv(f'data/cell_line/{cell_line}.csv')