# scASTRAL
Single-cell Afatinib Response of Triple Negative Cells

a pipeline for triple negative single cell data afatinib drug response prediction.
It includes a contrastive autoencoder, for dimensionality reduction and
feature extraction and a svm for the classification

----------------------------------
pipeline is provided as a pickled scikit-learn compatible estimator to be
easily integrated seamlessly in any python workflow.
It can be used with the following methods:
- predict(X)
- transform(X)
- predict_proba(X)
--------------------------------------------------
### project structure:

- data
  - get_data.py: script to download experiment data
  - afatinib.csv: afatinib drug response data
  - signature: gene signature
- models
  - scASTRAL_pipeline.sk: scikit-learn compatible estimator for scastral
  classification and feature extraction
- scastral
  - network.py: torch modules for scastral
  - utils.py:  utilities for loading and filtering data
  - preprocessing.py: scikit-learn compatible Transformers for count normalization
- train_model.ipynb: jupyter notebook illustrating model training
- validate_model.ipynb: jupyter model illustrating model validation