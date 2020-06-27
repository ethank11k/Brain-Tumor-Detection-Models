from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

xb = xgb.XGBClassifier()

parameter_selection = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
        
grid_search = GridSearchCV(estimator=xb, param_grid=parameter_selection, scoring = 'roc_auc', n_jobs = 10,cv = 10, verbose=True)

grid_search.fit(feat_train_res,y_train)

print(grid_search.best_estimator_.score(feat_train_res,y_train))
print(grid_search.best_estimator_.score(feat_val_res,y_val))
print(grid_search.best_estimator_.score(feat_test_res,y_test))