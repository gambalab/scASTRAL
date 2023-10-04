import pandas as pd

from scastral.preprocessing import CountPerMilionNormalizer, GfIcfTransformer
from scastral.utils import load_adata

signature = pd.read_csv('data/signature_374.csv')
signature = list(signature['ensembl_gene_id'])

input_path = ''  # expression matrix in 10x format
output_file = ''  # output csv file path
ensemblid_key = ''  # the colname used in adata for gene ensemblid

adata = load_adata(input_path,
                   mat_file='matrix.mtx.gz',
                   barc_file='barcodes.tsv.gz',
                   feat_file='features.tsv.gz')  # reading data
# filtering
X = adata.X.toarray()
X = X[X.sum(axis=1) > 5000, :]
# normalizing
X = pd.DataFrame(CountPerMilionNormalizer(norm_factors='edgeR').fit_transform(X),
                 columns=[gene.split('.')[0] for gene in adata.var[ensemblid_key]])
# removing duplicates
X = X.loc[:, ~X.columns.duplicated()]
# adding missing features
for gene in signature:
    if gene not in X.columns:
        X[gene] = 0

# subset to signature
X = X.loc[:, signature]

# gficf transform
gficf = GfIcfTransformer()
gficf.fit(X)

# write result
X = pd.DataFrame(gficf.transform(X), index=X.index, columns=X.columns)
X.to_csv(output_file)
