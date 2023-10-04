import os
from random import seed

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from scastral import *
from scastral.preprocessing import CountPerMilionNormalizer, GfIcfTransformer
from scastral.utils import load_adata

seed(1234)
np.random.seed(1234)
torch.random.manual_seed(1234)

train_set_times = ['data/train_set/CTRL',
                   'data/train_set/DTPE',
                   'data/train_set/T072',
                   'data/train_set/T144',
                   'data/train_set/T216']

signatures = ['data/signature_374.csv',
              'data/signature_549.csv',
              'data/signature_856.csv']

labels = pd.read_csv('data/train_set/cell_meta_dead_surv.tsv', header=None)[0].to_numpy()
response = pd.read_csv('data/validation_set/afatinib.csv')
response['cl'] = [p.replace('-', '') for p in response['Cell line']]
response['log(IC50)'] = np.log2(response['IC50'])
response = response.loc[:, ['cl', 'log(IC50)']]
response['score'] = np.nan

result = {'signature': [], 'time': [], 'SCC': [], 'pval': []}
for t in tqdm(train_set_times):
    for sign in signatures:
        print('preprocessing train set')
        # PREPARE TRAIN SET ===========================================
        signature = pd.read_csv(sign)
        signature = list(signature['ensembl_gene_id'])
        adata = load_adata(t)

        # filtering
        train = adata.X.toarray()
        cell_filter = (train.sum(axis=1) > 5000).flatten()
        train = train[cell_filter, :]
        y = labels[cell_filter]

        # normalization
        X = pd.DataFrame(CountPerMilionNormalizer(norm_factors='edgeR').fit_transform(train),
                         columns=[gene.split(' ')[0] for gene in adata.var[0]])

        # removing duplicates
        X = X.loc[:, ~X.columns.duplicated()]
        # adding missing features
        for gene in signature:
            if gene not in X.columns:
                X[gene] = 0
        # write preprocessed_pretfidf data
        X = X.loc[:, signature]
        X = GfIcfTransformer().fit_transform(X)

        print('training model')
        # TRAIN MODEL ==================================================
        clf = SVC(C=100, kernel=cosine_similarity, probability=True, random_state=123)
        model = network.SCAstral(max_epochs=250, patience=20, batch_size=128,
                                 input_size=X.shape[1], hidden_size=64, latent_size=32,
                                 alfa=1, mu=0.1, theta=1.5, lr=0.0001, verbose=True, path='models/weights.pt',
                                 predictor=clf)
        pipe = Pipeline([('feature_extraction', model), ('clf', clf)])

        pipe.fit(X, labels)

        # VALID MODEL ===================================================
        print('validation model')
        resp = response.copy()
        for cl in tqdm(os.listdir('data/rawdata')):
            if cl in list(response['cl']):
                adata = load_adata(f"data/rawdata/{cl}",
                                   mat_file='matrix.mtx.gz',
                                   barc_file='barcodes.tsv.gz',
                                   feat_file='features.tsv.gz')
                # filtering
                X = adata.X.toarray()
                X = X[X.sum(axis=1) > 5000, :]
                # normalizing
                X = pd.DataFrame(CountPerMilionNormalizer(norm_factors='edgeR').fit_transform(X),
                                 columns=[gene.split('.')[0] for gene in adata.var[0]])
                # removing duplicates
                X = X.loc[:, ~X.columns.duplicated()]
                # adding missing features
                for gene in signature:
                    if gene not in X.columns:
                        X[gene] = 0
                # write preprocessed_pretfidf data
                X = X.loc[:, signature]
                X = GfIcfTransformer().fit_transform(X)

                response.loc[response['cl'] == cl, 'score'] = pipe.predict(X).mean()

        response = response.dropna()
        test = pearsonr(response['log(IC50)'], response['score'])
        result['time'].append(t.split('/')[-1])
        result['signature'].append(sign.split('_')[-1].split('.')[0])
        result['SCC'].append(test.statistic)
        result['pval'].append(test.pvalue)

pd.DataFrame.from_dict(result).to_csv('data/rev1_results.csv')
