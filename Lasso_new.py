def Lasso_new(self, X, y):
    # parameter tuning
    alphaarr = [1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13,
                1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
                1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    print("++Sumit+ code-- LAsso")

    ##LASSO CODE

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # , alpha=float(alpha)
    best_result = []
    kfold = model_selection.KFold(n_splits=10)
    for alpha in alphaarr:
        clf = SGDClassifier(loss="log", penalty="l1", alpha=float(alpha))
        ress = clf.fit(X, y)
        model = fs.SelectFromModel(ress, prefit=True)
        X_new = model.transform(X)

        cv_results = model_selection.cross_val_score(SGDClassifier(loss="log", penalty="l1", alpha=float(alpha)), X_new, y,
                                                     cv=kfold, scoring='accuracy')
        ## check if train data can be scaled
        # print("cv_results",cv_results.mean(),'alpha --',alpha)
        best_result.append(cv_results.mean())
    bob = best_result.index(max(best_result))
    print('bob--', bob, 'best_result-', best_result[bob], 'alpha-', alphaarr[bob])