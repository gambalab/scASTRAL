from random import choice

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from torch import nn, Tensor
from torch.nn.functional import mse_loss, relu
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        return x


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
        x = self.relu(x)
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

    def get_sparsity(self, x):  # calculate the sparsity penalty
        loss = 0
        values = x
        for model_child in list(self.children()):
            values = relu((model_child(values)))
            loss += torch.mean(torch.abs(values))
        return loss


class SCAstral(BaseEstimator, TransformerMixin):
    """
    Contrastive Autoencoder
    """

    def __init__(self, input_size=200, hidden_size=64, latent_size=32, batch_size=32, max_epochs=200,
                 lr=.0001, sigma=.5, mu=.5, theta=.5, alfa=1, patience=np.inf, path='scae.pt', verbose=False,
                 predictor=None, metric=accuracy_score):

        """
        scAstral constructor

        :param input_size: input layer size
        :param hidden_size: hidden layer size
        :param latent_size: latent space size
        :param batch_size:  batch size for training
        :param max_epochs:  maximum number of train epochs
        :param lr: learning rate
        :param sigma: coefficient for sparsity loss
        :param mu:  margin for contrastive loss
        :param theta:  coefficient for contrastive loss
        :param alfa:
        :param patience:  maximum number of epochs with no improvement
        :param path:  path where to save model
        :param verbose:  print info about training
        :param predictor:  predictor for the latent space
        :param metric:  metric to compute on latent space
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.e_net = Encoder(input_size, hidden_size, latent_size).to(self.device)
        self.d_net = Decoder(input_size, hidden_size, latent_size).to(self.device)
        self.autoencoder = PairsAutoEncoder(self.e_net, self.d_net).to(self.device)

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.alfa = alfa

        self.valid = None
        self.valid_y = None
        self.training_summary = None
        self.path = path
        self.verbose = verbose
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.ae_train = None
        self.ae_valid = None
        self.cl_valid = None
        self.cl_valid_y = None

        self.predictor = predictor
        self.metric = metric

    def transform(self, X, y=None):
        """
        :param X: input
        :param y: only for compatibility
        :return:
        """
        return self.e_net(torch.tensor(X, device=self.device, dtype=torch.float32)).detach().cpu().numpy()

    def set_valid_data(self, X_valid, y_valid):
        self.valid = X_valid
        self.valid_y = y_valid

    def reset_weights(self):
        for layer in self.autoencoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def fit(self, X, y):

        assert (self.valid is not None)

        self.reset_weights()

        # data structure that keep training curve
        self.training_summary = {'epoch': np.array(range(self.max_epochs)),
                                 'train_loss': np.zeros(self.max_epochs),
                                 'valid_loss': np.zeros(self.max_epochs)}
        if self.predictor is not None:
            self.training_summary['metric'] = np.zeros(self.max_epochs)

        # initialize dataloader
        train_dl = DataLoader(ContrastiveAEDataset(X, y), batch_size=self.batch_size)
        valid_dl = DataLoader(ContrastiveAEDataset(self.valid, self.valid_y),
                              batch_size=self.batch_size)

        # initialize optimizer and loss functions
        optim = Adam(self.autoencoder.parameters(), lr=self.lr)
        contrastive_loss = nn.CosineEmbeddingLoss(margin=self.mu, reduction='sum')

        min_loss = np.inf
        max_acc = -np.inf
        no_improv_epoch = 0
        stop = False
        best_epoch = -1

        # training for each epoch
        for epoch in range(self.max_epochs):
            if stop:
                print(f"best epoch {best_epoch + 1}")
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
                loss = self.alfa * mse_loss(rec, X1) + \
                    self.sigma * self.autoencoder.get_sparsity(X1) + \
                    self.theta * contrastive_loss(emb1, emb2, flag)  # compute loss

                loss.backward()  # backprop
                optim.step()

                # update summary
                self.training_summary['train_loss'][epoch] += float(loss)
            self.training_summary['train_loss'][epoch] /= len(train_dl)

            """
            valid autoencoder
            """
            with torch.no_grad():

                self.autoencoder.eval()
                for batch_idx, (X1, X2, flag) in tqdm(enumerate(valid_dl), total=len(valid_dl),
                                                      disable=not self.verbose):
                    optim.zero_grad()

                    emb1, emb2, rec = self.autoencoder(X1, X2)  # forward
                    loss = self.alfa * mse_loss(rec, X1) + \
                        self.sigma * self.autoencoder.get_sparsity(X1) + \
                        self.theta * contrastive_loss(emb1, emb2, flag)  # compute loss

                    # update summary
                    self.training_summary['valid_loss'][epoch] += float(loss)
                self.training_summary['valid_loss'][epoch] /= len(valid_dl)

                # test accuracy wrt a predictor
                if self.predictor is not None:

                    """ 
                    test predictor accuracy 
                    """
                    train_emb = self.transform(X)
                    test_emb = self.transform(self.valid)
                    self.predictor.fit(train_emb, y)
                    pred = self.predictor.predict(test_emb)
                    self.training_summary['metric'][epoch] = self.metric(self.valid_y, pred)
                    if self.verbose:
                        print(f"metric: {self.training_summary['metric'][epoch]:.4f}")

                # early stopping
                if self.predictor is None and self.training_summary['valid_loss'][epoch] < min_loss:
                    if self.verbose:  # valiation loss based
                        print('updating')
                    torch.save({'model_ae_state': self.autoencoder.state_dict()}, self.path)
                    no_improv_epoch = 0
                    best_epoch = epoch
                    min_loss = self.training_summary['valid_loss'][epoch]
                elif self.predictor is not None and self.training_summary['metric'][epoch] > max_acc:
                    if self.verbose:  # accuracy based
                        print('updating')
                    torch.save({'model_ae_state': self.autoencoder.state_dict()}, self.path)
                    no_improv_epoch = 0
                    best_epoch = epoch
                    max_acc = self.training_summary['metric'][epoch]
                else:
                    no_improv_epoch += 1
                    if no_improv_epoch > self.patience:
                        stop = True

                # log improvements
                if self.verbose:
                    print(f"train loss:{self.training_summary['train_loss'][epoch]:.4f}")
                    print(f"valid loss:{self.training_summary['valid_loss'][epoch]:.4f}")

        # restore best weights
        checkpoint = torch.load(self.path)
        self.autoencoder.load_state_dict(checkpoint['model_ae_state'])
        return self
