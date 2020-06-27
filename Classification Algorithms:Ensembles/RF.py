from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
n_estimators = [100, 300, 500]
max_depth = [3, 5, 7, 10]
min_samples_split = [2, 5, 10, 100]
min_samples_leaf = [1, 2, 5, 10] 

parameter_selection = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(RF, parameter_selection, cv = 3, verbose = 1, n_jobs = -1)
gridF.fit(feat_train_res, y_train)

print(gridF.best_estimator_.score(feat_train_res,y_train))
print(gridF.best_estimator_.score(feat_val_res,y_val))
print(gridF.best_estimator_.score(feat_test_res,y_test))