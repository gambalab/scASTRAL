from random import choice
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def roc_auc_scorer(data, label, predictor):
    return roc_auc_score(label, predictor.predict_proba(data)[:, 1])


def accuracy_scorer(data, label, predictor):
    return accuracy_score(label, predictor.predict(data))


class ContrastiveAEDataset(Dataset):
    """
    Dataset for scAstral
    """

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.pairs = []
        for i in range(len(label)):
            self.pairs.append((i, 1))  # add a positive pair
            self.pairs.append((i, -1))  # add a negative pair
        """ for i in range(len(label)):
            for j in range(i + 1, len(label)):
                self.pairs.append((i, j, 1 if label[i] == label[j] else -1))"""



    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        #i, j, label = self.pairs[idx]

        i, label = self.pairs[idx]
        curr_label = self.label[i]
        if label == 1:  # random sampling
            j = choice(np.argwhere(np.array(self.label == curr_label)))[0]
        else:
            j = choice(np.argwhere(np.array(self.label != curr_label)))[0]

        return torch.tensor(self.data[i, :], device=self.device, dtype=torch.float32), \
            torch.tensor(self.data[j, :], device=self.device, dtype=torch.float32), \
            torch.tensor(label, device=self.device, dtype=torch.float32)


class Encoder(nn.Module):
    """
    Encoder network with 3 hidden layer
    """

    def __init__(self, input_size=3000, hidden_size=512, latent_size=32):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu1 = nn.ReLU()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, latent_size)
        # self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class Decoder(nn.Module):
    """
    Decoder network with 3 hidden layer
    """

    def __init__(self, output_size=3000, hidden_size=512, latent_size=32):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PairsAutoEncoder(nn.Module):
    """
    an AutoEncoder that process 2 sample at the same time
    """

    def __init__(self, encoder, decoder):
        """
        initialize AutoEncoder.
        N.B. output size of encoder must be equal to input size of decoder,
             input size of encoder must be equal to output size of decoder

        :param encoder: an Encoder object
        :param decoder:  a Decoder object
        """
        super(PairsAutoEncoder, self).__init__()
        self.e_net = encoder
        self.d_net = decoder

    def forward(self, x1, x2):
        e1 = self.e_net(x1)
        e2 = self.e_net(x2)
        rec = self.d_net(e1)
        return e1, e2, rec

    def encode(self, x):
        return self.e_net(x)

    def decode(self, x):
        return self.d_net(x)


class SCAstral(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Contrastive Autoencoder
    """

    def __init__(self, input_size=200, hidden_size=64, latent_size=32, batch_size=32, min_epochs=5, max_epochs=200,
                 lr=.0001, eps=1e-8, weight_decay=0, mu=.5, theta=.5, alfa=1, patience=np.inf, path='scastral.pt',
                 verbose=False, early_stop_metric='accuracy',
                 predictor=None, eval_metrics=None, pct_valid=.25):
        """
        scAstral constructor

        :type min_epochs: minimum number of epochs
        :param input_size: input layer size
        :param hidden_size: hidden layer size
        :param latent_size: latent space size
        :param batch_size:  batch size for training
        :param max_epochs:  maximum number of train epochs
        :param lr: learning rate
        :param mu:  margin for contrastive loss
        :param theta:  coefficient for contrastive loss
        :param alfa: coefficient for reconstruction loss
        :param patience:  maximum number of epochs with no improvement
        :param path:  path where to save model
        :param verbose:  print info about training
        :param predictor:  predictor for the latent space
        :param early_stop_metric: the key of the metric to be used for early stopping
        :param eval_metrics:  dict of pairs 'metric_name': scorer
        :param eps: Adam epsilon parameter
        :param weight_decay: Adam weight decay parameter
        """

        # network building ================================================
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.e_net = Encoder(input_size, hidden_size, latent_size).to(self.device)
        self.d_net = Decoder(input_size, hidden_size, latent_size).to(self.device)
        self.autoencoder = PairsAutoEncoder(self.e_net, self.d_net).to(self.device)

        # hyperparameters===============================================

        # adam
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps

        # loss function
        self.mu = mu
        self.theta = theta
        self.alfa = alfa

        # network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.batch_size = batch_size

        # Input/Output =================================

        self.training_summary = None
        self.verbose = verbose
        self.path = path

        # validation and early stopping=============================

        self.predictor = predictor
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience

        if eval_metrics is None:
            self.eval_metrics = {'accuracy': accuracy_scorer}
        else:
            self.eval_metrics = eval_metrics

        if isinstance(early_stop_metric, str) and early_stop_metric in eval_metrics.keys():
            self.early_stop_metric = early_stop_metric  # set early stopping metric
        else:
            self.patience = np.inf  # disable early stopping

    def transform(self, X, y=None):
        """
        :param X: input
        :param y: only for sklearn compatibility
        :return: embedding of input data as numpy array
        """
        return self.e_net(
            torch.tensor(X, device=self.device, dtype=torch.float32)
        ).detach().cpu().numpy()

    def predict(self, X, y=None):
        """
        :param X: input
        :param y: only for sklearn compatibility
        :return: prediction
        """
        return self.predictor.predict(self.transform(X))

    def predict_proba(self, X, y=None):
        """
        :param X: input
        :param y: only for sklearn compatibility
        :return: probability for each class
        """
        return self.predictor.predict_proba(self.transform(X))

    def predict_log_proba(self, X, y=None):
        """
        :param X: input
        :param y: only for sklearn compatibility
        :return: log probability for each class
        """
        return self.predictor.predict_log_proba(self.transform(X))

    def fit(self, X, y, X_test=None, y_test=None):
        """

        :param X: train data
        :param y: train labels
        :param X_test: test data
        :param y_test: test labels
        :return:  fitted estimator
        """

        # data structure that keep training curve
        self.training_summary = {'epoch': np.array(range(self.max_epochs)),
                                 'train_loss': np.zeros(self.max_epochs)}

        for metric in self.eval_metrics.keys():
            self.training_summary[metric] = np.zeros(self.max_epochs)

        # initialize dataloader
        train_dl = DataLoader(ContrastiveAEDataset(X, y), batch_size=self.batch_size)

        # initialize optimizer and loss functions
        optim = Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
        contrastive_loss = nn.CosineEmbeddingLoss(margin=self.mu, reduction='mean')

        # initialize variables for early stopping
        max_score = -np.inf
        no_improv_epoch = 0
        stop = False
        best_epoch = -1

        # training for each epoch
        for epoch in range(self.max_epochs):
            if stop:
                if self.verbose:
                    print(f"best epoch {best_epoch + 1} with {self.early_stop_metric}: {max_score}")
                break
            if self.verbose:
                print(f"-----epoch {epoch + 1}/{self.max_epochs}-----")

            """
            train autoencoder
            """
            self.autoencoder.train()
            for batch_idx, (X1, X2, flag) in tqdm(enumerate(train_dl), total=len(train_dl), disable=not self.verbose):
                optim.zero_grad()

                emb1, emb2, rec = self.autoencoder(X1, X2)  # forward
                loss = self.alfa * mse_loss(X1, rec) + \
                       self.theta * contrastive_loss(emb1, emb2, flag)  # compute loss

                loss.backward()  # backprop
                optim.step()

                # update summary
                self.training_summary['train_loss'][epoch] += float(loss)
            self.training_summary['train_loss'][epoch] /= len(train_dl)

            """
            valid autoencoder
            """
            self.autoencoder.eval()
            with torch.no_grad():
                emb_train = self.transform(X)
                emb_test = self.transform(X_test)
                self.predictor.fit(emb_train, y)

                for metric in self.eval_metrics.keys():  # compute eval metrics
                    self.training_summary[metric][epoch] = self.eval_metrics[metric](emb_test, y_test, self.predictor)

                if epoch <= self.min_epochs:
                    continue
                elif self.training_summary[self.early_stop_metric][epoch] >= max_score:  # early stopping
                    if self.verbose:
                        print('updating')
                    torch.save({'model_ae_state': self.autoencoder.state_dict()}, self.path)
                    no_improv_epoch = 0
                    best_epoch = epoch
                    max_score = self.training_summary[self.early_stop_metric][epoch]
                else:
                    no_improv_epoch += 1
                    if no_improv_epoch >= self.patience:
                        stop = True

            # log improvements
            if self.verbose:
                print(f"train loss:{self.training_summary['train_loss'][epoch]:.4f}")
                for metric in self.eval_metrics.keys():
                    print(f"{metric}: {self.training_summary[metric][epoch]:.4f}")

        # restore best weights
        checkpoint = torch.load(self.path, map_location='cpu')
        self.autoencoder.load_state_dict(checkpoint['model_ae_state'])

        emb_train = self.transform(X)
        emb_test = self.transform(X_test)

        self.predictor.fit(np.vstack([emb_train, emb_test]),
                           np.concatenate([y, y_test]))

        self.training_summary = pd.DataFrame.from_dict(self.training_summary)
        return self
