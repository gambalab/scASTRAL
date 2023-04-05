import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


def compute_scale_factors(readcounts, trim_m=0.3, trim_a=0.05):
    def scale_factor(x, ref):
        vsize = x.shape[0] // 2
        mask = x[vsize:].astype(bool)
        x = x[:vsize]

        norm_x = x / np.nansum(x)
        norm_ref = ref / np.nansum(ref)
        log_sample = np.log2(norm_x)
        log_ref = np.log2(ref)
        m = log_sample - log_ref
        a = (log_sample + log_ref) / 2

        perc_m = np.nanquantile(m, [trim_m, 1 - trim_m], method='nearest')
        perc_a = np.nanquantile(a, [trim_a, 1 - trim_a], method='nearest')

        mask |= (m < perc_m[0]) | (m > perc_m[1])
        mask |= (a < perc_a[0]) | (a > perc_a[1])

        w = ((1 - norm_x) / x) + ((1 - norm_ref) / ref)
        w = 1 / w

        w[mask] = 0
        m[mask] = 0
        w /= w.sum()
        return np.sum(w * m)

    readcounts = np.array(readcounts, dtype=float)
    q75_expr = np.apply_along_axis(lambda x: np.percentile(x[np.any(x != 0)], 75), axis=1, arr=readcounts)
    iref = np.argmin(np.abs(q75_expr - q75_expr.mean()))
    refsample = readcounts[iref, :]

    f = readcounts == 0
    f[:, f[iref]] = True
    readcounts[f] = np.nan

    funcin = np.concatenate((readcounts, f), axis=1)
    sf = np.apply_along_axis(lambda x: scale_factor(x, refsample), axis=1, arr=funcin)
    sf -= sf.mean()
    return np.exp2(sf)


class CountPerMilionNormalizer(TransformerMixin, BaseEstimator):
    def __init__(self, total=1e6, variance_stabilization=None):
        self.variance_stabilization = variance_stabilization
        self.total = total
        self.factors = None

    def fit(self, X, y=None):
        if self.variance_stabilization == True or self.variance_stabilization == 'tmm':
            self.factors = compute_scale_factors(X).reshape(-1, 1)
        return self

    def transform(self, X, y=None):
        X = normalize(X, norm='l1') * self.total
        if self.variance_stabilization is None:
            return X
        elif self.variance_stabilization == 'sqrt':
            return np.sqrt(X)
        elif self.variance_stabilization == 'log':
            return np.log(X + 1)
        else:
            return X / self.factors

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)


class GfIcfTransformer(BaseEstimator, TransformerMixin):
    """
    Gf-icf normalization provided as a scikit-learn transformer estimator
    """

    def __init__(self):
        self.tfidf = TfidfTransformer()

    def transform(self, X, y=None):
        return self.tfidf.transform(np.array(X)).toarray()

    def fit(self, X, y=None):
        self.tfidf.fit(np.array(X))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)
