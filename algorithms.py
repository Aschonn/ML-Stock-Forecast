from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, accuracy_score
import numpy as np
from sklearn import svm


def train_random_forest(X_train, y_train, X_test, y_test):
    
    # Create a new random forest classifier
    rf = RandomForestClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    #params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use gridsearch to test all values for n_estimators
    #rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
    # Fit model to training data
    rf.fit(X_train, y_train)
    
    # Save best model
    #rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    #print(rf_gs.best_params_)
    
    prediction = rf.predict(X_test)

#     print(classification_report(y_test, prediction))
#     print(confusion_matrix(y_test, prediction))
    
    return rf
    
#rf_model = _train_random_forest(X_train, y_train, X_test, y_test)



def train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 10)}
    
    # Use gridsearch to test all values for n_neighbors
    #knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
    # Fit model to training data
    knn.fit(X_train, y_train)
    
    # Save best model
    #knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    #print(knn_gs.best_params_)
    
#     prediction = knn.predict(X_test)

#     print(classification_report(y_test, prediction))
#     print(confusion_matrix(y_test, prediction))
    
    return knn
    
#knn_model = _train_KNN(X_train, y_train, X_test, y_test)




def train_GBT(X_train, y_train, X_test, y_test):
    
    clf = GradientBoostingClassifier()
    
    # Dictionary of parameters to optimize
    #params_gbt = {'n_estimators' :[150,160,170,180] , 'learning_rate' :[0.2,0.1,0.09] }
    
    # Use gridsearch to test all values for n_neighbors
    #grid_search = GridSearchCV(clf, params_gbt, cv=5)
    
    # Fit model to training data
    clf.fit(X_train, y_train)
    
    #gbt_best = grid_search.best_estimator_
    
    # Save best model
    #print(grid_search.best_params_)
    
    prediction = clf.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    
    return clf


#gbt_model = _train_GBT(X_train, y_train, X_test, y_test)

def ensemble_model(rf_model, knn_model, X_train, y_train, X_test, y_test):
    
    # Create a dictionary of our models
    estimators=[('knn', knn_model), ('rf', rf_model)]
    
    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft')
    
    #fit model to training data
    ensemble.fit(X_train, y_train)
    
    #test our model on the test data
    print(ensemble.score(X_test, y_test))
    
    prediction = ensemble.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    return ensemble
    
#ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test)