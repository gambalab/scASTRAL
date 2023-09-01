# scASTRAL
Single-cell Afatinib Response of Triple Negative Cells

a pipeline for triple negative single cell data afatinib drug response prediction.
It includes a contrastive autoencoder, for dimensionality reduction and
feature extraction and a svm for the classification

----------------------------------
pipeline is provided as a pickled scikit-learn compatible estimator to be
easily integrated seamlessly in any python workflow.
It can be used with the following methods:
- predict(X): predict the class of cell ( 1 means resistent)
- transform(X): get embedding of input vector
- predict_proba(X): get probability of each class
--------------------------------------------------
### project structure:

- data
  - afatinib.csv: afatinib drug response data
  - signature: gene signature
  - train_set: the preprocessed mdamb468 labeled cell line (only 374 genes)
  - cell line: contains preprocessed validation data
  - preprocessing script.py: script to preprocess data before giving in input to scASTRAL
- models
  - scASTRAL_pipeline.sk: scikit-learn compatible estimator for scastral
  classification and feature extraction
- scastral
  - network.py: torch modules for scastral
  - utils.py:  utilities for loading and filtering data
  - preprocessing.py: scikit-learn compatible Transformers for count normalization
- train_model.ipynb: jupyter notebook illustrating model training
- validate_model.ipynb: jupyter model illustrating model validation
- find_treshold.py: process to estimate confidence thresholds