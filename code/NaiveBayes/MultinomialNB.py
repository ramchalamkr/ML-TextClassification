import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


#execute this once the input data has been vectorised and for normal word frequency

x_1 = pickle.load(open('x_train_binary.pkl',"rb"))
y_1 = pickle.load(open('y_train.pkl',"rb"))

#include chisquare-based selection of features

class cv:
    def TestTrainSplit(self,X,y):
        k = np.random.permutation(X.shape[0])
        train_idx, test_idx = k[:TestSplit], k[TestSplit:]
        X_train,X_test = X[train_idx], X[test_idx]
        y_train,y_test = y[train_idx], y[test_idx]
        return X_train,X_test,y_train,y_test

obj = cv()

#X_train, X_test, y_train, y_test = obj.TestTrainSplit(x_1,y_1)
X_train, X_test, y_train, y_test = train_test_split(x_1,y_1,test_size=0.33, random_state=42)

pipeline = Pipeline([
    ('selectk', SelectKBest(score_func=chi2)), 
    ('nb', MultinomialNB()),        
])

parameters = {
    'selectk__k': (500,1000,2000,5000,10000,20000, 25000, 30000, 35000, 40000, 50000, 60000, 'all')
}

if __name__ == "__main__": #import for parallel processing with n_jobs
    gridSearchEst = GridSearchCV(pipeline, parameters, n_jobs=5, verbose=1, error_score=0, cv=5)
    gridSearchEst.fit(X_train, y_train)
    
    print("Best score: %0.3f" % gridSearchEst.best_score_)
    print("Best parameters set:")
    best_parameters = gridSearchEst.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    print(gridSearchEst.cv_results_['mean_test_score'])
    print(gridSearchEst.cv_results_['mean_train_score'])
    
pickle.dump(gridSearchEst.best_estimator_, open('fittedGridSearchEstimator.pkl',"wb"))    
pickle.dump(gridSearchEst.cv_results_, open('cvResults.pkl',"wb"))

