import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

model = pkl.load(open('models/scASTRAL_final_pipeline.pkl', 'rb'))

## %%
train = pd.read_csv('data/train_set.csv', index_col=0)
labels = np.array(train['label'])  # get labels

del train['label']

proba = model.predict_proba(train)[:, 1]

## %%


report = []
tresholds = np.linspace(0.1, 1, 20)
max_score = -np.inf
for t1 in tqdm(tresholds):
    for t2 in tresholds:
            mask = (proba <= t2) | (proba >= t1)
            pred = (proba[mask] >= t1).astype(int)
            discarted = (mask.shape[0] - pred.shape[0]) / mask.shape[0]
            pre = precision_score(labels[mask], pred)
            rec = recall_score(labels[mask], pred)
            score = (pre * rec * discarted) / (pre + rec + discarted)
            report.append({'pos': t1, 'neg': t2, 'score': score})
            if score > max_score:
                max_score = score
                best_tresholds = (t1, t2)
report = pd.DataFrame(report)

a1 = report.groupby('pos').mean()
a2 = report.groupby('neg').mean()
plt.plot(a1.index, a1['score'])
plt.plot(a2.index, a2['score'])
plt.show()
