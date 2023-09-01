




preprocessing_pipe = Pipeline([ ('gf-icf',preprocessing.GfIcfTransformer()),('scaler',MinMaxScaler())])
#clf = SVC(C=100,kernel=cosine_similarity,probability=True,class_weight='balanced',random_state=123)
clf = CalibratedClassifierCV(estimator=SVC(kernel='linear',probability=True,class_weight='balanced',random_state=123),cv=5,ensemble=True,n_jobs=-1)
scastral = network.SCAstral(max_epochs=50, patience=10,
                            input_size=train.shape[1], hidden_size=64, latent_size=32,
                            sigma=.5, mu=.5, theta = 1, lr=.0001,
                            predictor=clf, scorer=make_scorer(roc_auc_score,needs_threshold=True),
                            verbose=True, path='models/scmuta.pt')

X = preprocessing_pipe.fit_transform(train)
X = scastral.fit_transform(X,labels)
clf.fit(X,labels)

pipe = Pipeline([('preprocessing',preprocessing_pipe),('feature_extraction',scastral),('clf',clf)])