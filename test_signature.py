import random
from random import seed
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

import scastral.network
from scastral import *
#%%
path = 'data/validation_set/preprocessed'
train = pd.read_csv('data/train_set/preprocessed_trainset.csv')
train.set_index('Unnamed: 0', inplace=True, drop=True)  # set barcodes as index
labels = np.array(train['label'])  # get labels

del train['label']
signature = pd.read_csv('data/signature_374.csv')
signature = list(signature['ensembl_gene_id'])

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

response = pd.read_csv('data/validation_set/afatinib.csv')
response['cl'] = [p.replace('-', '') for p in response['Cell line']]
response['log(IC50)'] = np.log2(response['IC50'])
response = response.loc[:, ['cl', 'log(IC50)']]
response['score'] = np.nan

seed(1234)
np.random.seed(1234)
torch.random.manual_seed(1234)
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

for i in tqdm(range(1000)):
    signature = random.sample(list(train.columns), 374)
    X = train[signature].to_numpy()
    max_accuracy = -np.inf
    best_model = None
    for j, (train_index, test_index) in enumerate(cv.split(train, labels)):
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = labels[train_index]
        y_test = labels[test_index]

        clf = SVC(C=100, kernel=cosine_similarity, probability=True, random_state=123)
        model = network.SCAstral(max_epochs=250, patience=20, batch_size=32, min_epochs=-1,
                                 input_size=train.shape[1], hidden_size=64, latent_size=32,
                                 alfa=0.1, mu=0, theta=1, lr=0.0001, verbose=False, path='models/weights.pt',
                                 eval_metrics={'roc_auc': scastral.network.roc_auc_scorer,
                                               'accuracy': scastral.network.accuracy_scorer},
                                 early_stop_metric='accuracy',
                                 predictor=clf)

        model.fit(X_train, y_train, X_test, y_test)
        best_epoch = np.argmax(model.training_summary['accuracy'])
        accuracy = model.training_summary['accuracy'].iloc[best_epoch]
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = model

    result = response.copy()
    for cell_line in valid_set:
        df = pd.read_csv(f'data/validation_set/preprocessed/{cell_line}.csv', index_col=0)
        df = df[signature].to_numpy()
        proba = best_model.predict_proba(df)[:, 1]
        response.loc[response['cl'] == cell_line, 'score'] = np.mean(proba > .75)

    response = response.dropna()
    test = pearsonr(response['log(IC50)'], response['score'])
    results[i] = {'signature': signature, 'corr': test.statistic, 'pval': test.pvalue}
    json.dump(results, open('data/signature_valid_result.json', 'w'))
