import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


class CountPerMilionNormalizer(TransformerMixin, BaseEstimator):
    """
    cpm normalization provided as a scikit-learn transformer estimator
    """
    def __init__(self, log=False, factor=1e6):
        '''
        :param log: True to convert to log counts
        :param factor: scaling factor
        '''
        self.log = log
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = normalize(X, norm='l1') * self.factor
        if self.log:
            return np.log(X + 1)
        else:
            return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class GfIcfTransformer(BaseEstimator, TransformerMixin):
    """
    Gf-icf normalization provided as a scikit-learn transformer estimator
    """
    def __init__(self):
        self.tfidf = TfidfTransformer()

    def transform(self, X, y=None):
        return self.tfidf.transform(X).toarray()

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self

