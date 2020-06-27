from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()

parameter_selection = {
    'n_neighbors': [3,5,11,19],
    'weights': ['uniform','distance'],
    'metric': ['euclidean','manhatten']
}

gs_knn = GridSearchCV(knc, param_grid = parameter_selection, n_jobs = 1)

gs_knn.fit(feat_train_res,y_train)

print(gs_knn.best_estimator_.score(feat_train_res,y_train))
print(gs_knn.best_estimator_.score(feat_val_res,y_val))
print(gs_knn.best_estimator_.score(feat_test_res,y_test))