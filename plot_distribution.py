from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import scastral.network
from scastral import *

path = 'data/validation_set/preprocessed'
train = pd.read_csv('data/train_set/preprocessed_trainset.csv')
train.set_index('Unnamed: 0', inplace=True, drop=True)  # set barcodes as index
labels = np.array(train['label'])  # get labels

del train['label']
signature = pd.read_csv('data/signature_374.csv')
signature = list(signature['ensembl_gene_id'])

X = pd.DataFrame(train,
                 columns=train.columns,
                 index=train.index)

train = X.loc[:, signature].to_numpy()  # subset to signature

seed(1234)
np.random.seed(1234)
torch.random.manual_seed(1234)

X_train, X_test, y_train, y_test = train_test_split(train, labels, train_size=.7, random_state=123)

clf = SVC(C=100, kernel=cosine_similarity, probability=True, random_state=123)
model = network.SCAstral(max_epochs=250, patience=20, batch_size=32, min_epochs=-1,
                         input_size=train.shape[1], hidden_size=64, latent_size=32,
                         alfa=0.1, mu=0, theta=1, lr=0.0001, verbose=True, path='models/weights.pt',
                         eval_metrics={'roc_auc': scastral.network.roc_auc_scorer,
                                       'accuracy': scastral.network.accuracy_scorer},
                         early_stop_metric='accuracy',
                         predictor=clf)

model.fit(X_train, y_train, X_test, y_test)

transformed = model.transform(X_train)
mask = np.array(y_train, dtype=bool)

# compute cosine distance after transformation
sxsa = cosine_distances(transformed[mask, :])  # survived on survived
sxda = cosine_distances(transformed[mask, :], transformed[~mask, :])  # survived on dead
dxda = cosine_distances(transformed[~mask, :])  # dead on dead

np.fill_diagonal(sxsa, np.nan)  # remove diagonal elements
np.fill_diagonal(sxda, np.nan)  # remove diagonal elements
np.fill_diagonal(dxda, np.nan)  # remove diagonal elements

inter_a = sxda.flatten()
intra_a = np.concatenate((sxsa.flatten(), dxda.flatten()))

legend_elements = [Patch(facecolor='peachpuff', edgecolor='orange',
                         label='inter'), Patch(facecolor='lightblue', edgecolor='royalblue',
                                               label='intra')]

# plot distribution after transformation
sns.kdeplot(intra_a, color='royalblue', fill=True, legend='intra')
sns.kdeplot(inter_a, color='orange', fill=True, legend='inter')
plt.xlabel('cosine distance')
plt.legend(handles=legend_elements, loc='upper right')
plt.savefig('distribution_after.pdf')
plt.show()
