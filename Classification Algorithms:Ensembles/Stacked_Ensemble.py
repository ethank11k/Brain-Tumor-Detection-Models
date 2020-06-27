from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

#SVM KNN: ('s2', clf_res.best_estimator_), ('s', gs_knn.best_estimator_)
#SVM XGB: ('s2', clf_res.best_estimator_), ('s4',grid_search.best_estimator_) 
#SVM Random: ('s2', clf_res.best_estimator_), ('s3', gridF.best_estimator_)
#KNN XGB: ('s', gs_knn.best_estimator_), ('s4',grid_search.best_estimator_) 
#KNN Random: ('s', gs_knn.best_estimator_), ('s3', gridF.best_estimator_)
#XGB Random: ('s4',grid_search.best_estimator_), ('s3', gridF.best_estimator_)
#SVM KNN XGB: ('s', gs_knn.best_estimator_), ('s2', clf_res.best_estimator_), ('s4',grid_search.best_estimator_)
#SVM KNN Random: ('s', gs_knn.best_estimator_), ('s2', clf_res.best_estimator_), ('s3', gridF.best_estimator_)
#SVM XGB Random: ('s2', clf_res.best_estimator_), ('s3', gridF.best_estimator_), ('s4',grid_search.best_estimator_) 
#KNN XGB Random: ('s3', gridF.best_estimator_), ('s4',grid_search.best_estimator_), ('s', gs_knn.best_estimator_)
#SVM KNN XGB Random: ('s2', clf_res.best_estimator_), ('s3', gridF.best_estimator_), ('s4',grid_search.best_estimator_), ('s', gs_knn.best_estimator_)
estimators_res = [
  ('s2', clf_res.best_estimator_), ('s3', gridF.best_estimator_), ('s4',xb), ('s', gs_knn.best_estimator_)

]

stacked = StackingClassifier(
     estimators=estimators_res, final_estimator=LogisticRegression()
)

stacked.fit(feat_train_res, y_train)

print(stacked.score(feat_train_res,y_train))
print(stacked.score(feat_val_res,y_val))
print(stacked.score(feat_test_res,y_test))

