import numpy as np
from scipy.io import mmread
import pandas as pd
import anndata as ad


def filter_data(data, min_cell_per_gene_pct=15, min_nonz_expr=1.12, min_detect_gene=.05, min_umi_per_cell=5000):
    """
    :param data: data object castable to ndarray
    :param min_cell_per_gene_pct: threshold for gene filter. genes detected in less then this % of cells
    will be removed
    :param min_nonz_expr: minimum non-zero mean expression for genes
    :param min_detect_gene:  minimum number of detected genes
    :param min_umi_per_cell: cells with less than this number of UMI will be removed
    :return: two boolean array that can be used as filters
    """
    data = np.array(data)
    cell_count_cutoff = np.log10(min_cell_per_gene_pct)  # genes detected in less than 15% will be excluded
    nonz_mean_cutoff = np.log10(
        min_nonz_expr)  # if cutoff2 < gene < cutoff, select only genes with nonzero mean expression > 1.12
    cell_count_cutoff2 = np.log10(
        data.shape[0] * min_detect_gene + 1e-7)  # genes detected in at least this will be included
    cells_per_gene = (data > 0).sum(axis=0) + 1e-7  # computing count of cells per gene
    nonz_mean = data.sum(axis=0) / cells_per_gene + 1e-7  # computing mean expression

    # switch to log10
    cells_per_gene = np.log10(cells_per_gene)
    nonz_mean = np.log10(nonz_mean)

    # creating filter as a boolean array
    gene_filter = np.array(cells_per_gene > cell_count_cutoff2) | (
            np.array(cells_per_gene > cell_count_cutoff)
            & np.array(nonz_mean > nonz_mean_cutoff))

    cell_filter = data.sum(axis=1) > min_umi_per_cell  # computing the umi count per cell

    return cell_filter, gene_filter


# load an adata file
def load_adata(data_dir, mat_file='matrix.mtx', barc_file='barcodes.tsv', feat_file='features.tsv'):
    """
    load an Adata Object 10x format
    :param data_dir:  directory where count data are
    :param mat_file: mtx file containing matrix
    :param barc_file:  tsv containing barcodes
    :param feat_file:  tsv containing features
    :return: anndata object
    """
    matrix = mmread(f'{data_dir}/{mat_file}')  # load mtx
    barcodes = pd.read_csv(f'{data_dir}/{barc_file}', sep='\t', header=None)  # load barcodes
    features = pd.read_csv(f'{data_dir}/{feat_file}', sep='\t', header=None)  # load features

    return ad.AnnData(X=matrix.T,  # initialize anndata
                      obs=barcodes,
                      var=features,
                      dtype=matrix.dtype)
