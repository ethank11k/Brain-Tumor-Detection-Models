import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

parameter_selection = [
  {'C': Cs, 'kernel': ['linear']},
  {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']},
  {'C': Cs, 'gamma': gammas, 'kernel': ['poly']},
  {'C': Cs, 'gamma': gammas, 'kernel': ['sigmoid']},
]

clf_res = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_selection, n_jobs=-1)

clf_res.fit(feat_train_res, y_train)

print(clf_res.best_estimator_.score(feat_train_res,y_train))
print(clf_res.best_estimator_.score(feat_val_res,y_val))
print(clf_res.best_estimator_.score(feat_test_res,y_test))
