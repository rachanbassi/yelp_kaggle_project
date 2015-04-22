__author__ = 'alicebenziger'

from data_encoding import train_test
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LinearRegression,Ridge


def rsmle_(predicted,actual):
    """
    The function computes a Root Mean Squared Logarithmic Error
    between the actual and predicted response variables
    """
    actual = np.exp(actual)-1
    predicted = np.exp(predicted)-1
    return np.sqrt(np.mean((pow(np.log(predicted+1) - np.log(actual+1),2))))


def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None):
    """
    :param estimator: sklearn regressor object
    :param title: Title of the curve
    :param X: predictors
    :param y: response
    :param param_name: parameter of the regression obj to do cross validation on, ex:Number of trees for RF
    :param param_range: range of values of the parameter
    :param ylim:
    :return: plots the validation curve for the parameter
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("RMSLE")
    rsmle_score = make_scorer(rsmle_,greater_is_better=True)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring=rsmle_score, n_jobs=1)
    print "cross validation done...plotting the graph"
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    :param estimator: sklearn regressor object
    :param title: title of the curve
    :param X: predictors
    :param y: response
    :param ylim:
    :param cv: number of cross validation folds
    :param n_jobs: for parallel computing
    :param train_sizes: a list of the training sizes
    :return: plots the learning curve
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("RMSLE")
    rsmle_score = make_scorer(rsmle_,greater_is_better=True)
    train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = rsmle_score)
    print "plotting learning curve.."
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


def random_forest_regressor(train_x, train_y, pred_x, review_id, v_curve=False, l_curve=False, get_model=True):
    """
    :param train_x: train
    :param train_y: text
    :param pred_x: test set to predict
    :param review_id: takes in a review id
    :param v_curve: run the code for validation curve
    :param l_curve: run the code for learning curve
    :param get_model: run the code
    :return:the predicted values,learning curve, validation curve
    """
    rf = RandomForestRegressor(n_estimators=20,criterion='mse',max_features='auto', max_depth=10)
    if get_model:
        print "Fitting RF..."
        rf.fit(train_x, np.log(train_y+1))
        print rf.score(train_x, np.log(train_y+1))
        rf_pred = np.exp(rf.predict(pred_x))-1.0
        Votes = rf_pred[:,np.newaxis]
        Id = np.array(review_id)[:,np.newaxis]
        submission_rf = np.concatenate((Id,Votes),axis=1)
        # create submission csv for Kaggle
        np.savetxt("submission_rf.csv", submission_rf,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')
    # plot validation and learning curves
    if v_curve:
        train_y = np.log(train_y+1.0)
        plot_validation_curve(RandomForestRegressor(), "Random Forest: Validation Curve(No: of trees)", train_x,train_y,'n_estimators',[5,10,20,50,100])
    if l_curve:
        train_y = np.log(train_y+1.0)
        plot_learning_curve(RandomForestRegressor(), "Random Forest: Learning Curve", train_x,train_y)


def ada_boost_regressor(train_x, train_y, pred_x, review_id, v_curve=False, l_curve=False, get_model=True):
    """
    :param train_x: train
    :param train_y: text
    :param pred_x: test set to predict
    :param review_id: takes in a review id
    :param v_curve: run the model for validation curve
    :param l_curve: run the model for learning curve
    :param get_model: run the model
    :return: the predicted values,learning curve, validation curve
    """
    ada = AdaBoostRegressor(n_estimators=5)
    if get_model:
        print "Fitting Ada..."
        ada.fit(train_x, np.log(train_y+1))
        ada_pred = np.exp(ada.predict(pred_x))-1
        Votes = ada_pred[:,np.newaxis]
        Id = np.array(review_id)[:,np.newaxis]
        # create submission csv for Kaggle
        submission_ada= np.concatenate((Id,Votes),axis=1)
        np.savetxt("submission_ada.csv", submission_ada,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')
    # plot validation and learning curves
    if l_curve:
        print "Working on Learning Curves"
        plot_learning_curve(AdaBoostRegressor(), "Learning curve: Adaboost", train_x, np.log(train_y+1.0))
    if v_curve:
        print "Working on Validation Curves"
        plot_validation_curve(AdaBoostRegressor(), "Validation Curve: Adaboost", train_x, np.log(train_y+1.0),
                              param_name="n_estimators", param_range=[2, 5, 10, 15, 20, 25, 30])


def gradient_boosting_regressor(train_x, train_y, pred_x, review_id, v_curve=False, l_curve=False, get_model=True):
    """
    :param train_x: train
    :param train_y: text
    :param pred_x: test set to predict
    :param review_id: takes in a review id
    :param v_curve: run the model for validation curve
    :param l_curve: run the model for learning curve
    :param get_model: run the model
    :return:the predicted values,learning curve, validation curve
    """
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=7, random_state=7)
    if get_model:
        print "Fitting GBR..."
        gbr.fit(train_x, np.log(train_y+1))
        gbr_pred = np.exp(gbr.predict(pred_x))- 1
        #dealing with
        for i in range(len(gbr_pred)):
            if gbr_pred[i] < 0:
                gbr_pred[i] = 0
        Votes = gbr_pred[:, np.newaxis]
        Id = np.array(review_id)[:, np.newaxis]
        submission_gbr = np.concatenate((Id,Votes),axis=1)
        np.savetxt("submission_gbr.csv", submission_gbr,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')
    # plot validation and learning curves
    if v_curve:
        print "Working on Validation Curves"
        plot_validation_curve(GradientBoostingRegressor(), "Validation Curve: GBR", train_x, np.log(train_y+1.0),
                              param_name="n_estimators", param_range=[5, 20, 60, 100, 150, 200])
    if l_curve:
        print "Working on Learning Curves"
        plot_learning_curve(GradientBoostingRegressor(), "Learning Curve: GBR", train_x, np.log(train_y+1.0))


def linear_regression(train_x, train_y, pred_x, review_id, v_curve=False, l_curve=False, get_model=True):
    """
    :param train_x: train
    :param train_y: text
    :param pred_x: test set to predict
    :param review_id: takes in a review id
    :param v_curve: run the model for validation curve
    :param l_curve: run the model for learning curve
    :param get_model: run the model
    :return:the predicted values,learning curve, validation curve
    """
    lin = LinearRegression(normalize=True)
    if get_model:
        print "Fitting Linear..."
        lin.fit(train_x, np.log(train_y+1))
        gbr_pred = np.exp(lin.predict(pred_x))- 1
        for i in range(len(gbr_pred)):
            if gbr_pred[i] < 0:
                gbr_pred[i] = 0
        Votes = gbr_pred[:,np.newaxis]
        Id = np.array(review_id)[:,np.newaxis]
        submission_lin= np.concatenate((Id,Votes),axis=1)
        np.savetxt("submission_lin.csv", submission_lin,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')
    # plot validation and learning curves
    if v_curve:
        pass
    if l_curve:
        print "Working on Learning Curves"
        plot_learning_curve(LinearRegression(), "Learning Curve for Linear Regression", train_x, np.log(train_y+1.0))


def ridge_regression(train_x, train_y, pred_x, review_id, v_curve=False, l_curve=False, get_model=True):
    """
   :param train_x: train
   :param train_y: text
   :param pred_x: test set to predict
   :param review_id: takes in a review id
   :param v_curve: run the model for validation curve
   :param l_curve: run the model for learning curve
   :param get_model: run the model
   :return:the predicted values,learning curve, validation curve
   """
    lin = Ridge(alpha=0.5)
    if get_model:
        print "Fitting Ridge..."
        lin.fit(train_x, np.log(train_y+1))
        gbr_pred = np.exp(lin.predict(pred_x))- 1
        for i in range(len(gbr_pred)):
            if gbr_pred[i] < 0:
                gbr_pred[i] = 0
        Votes = gbr_pred[:, np.newaxis]
        Id = np.array(review_id)[:, np.newaxis]
        submission_lin= np.concatenate((Id,Votes),axis=1)
        np.savetxt("submission_ridge.csv", submission_lin,header="Id,Votes", delimiter=',',fmt="%s, %0.2f", comments='')
    if v_curve:
        print "Working on Validation Curves"
        plot_validation_curve(Ridge(), "Validation Curve for Ridge Regression", train_x, np.log(train_y+1.0),
                              param_name="alpha", param_range=[0.1,0.2,0.5,1,10])
    if l_curve:
        print "Working on Learning Curves"
        plot_learning_curve(Ridge(), "Learning Curve for Linear Regression", train_x, np.log(train_y+1.0))

if __name__ == '__main__':
    ## fetching training data to train the model on and testing data to predict the results
    train_x, train_y, pred_x, train_x_norm, pred_x_norm, review_id = train_test()
    print "data fetched..."
    ## ML models
    random_forest_regressor(train_x, train_y, pred_x, review_id, get_model=True)
    gradient_boosting_regressor(train_x, train_y, pred_x, review_id, get_model=True)
    ada_boost_regressor(train_x, train_y, pred_x, review_id, get_model=True)
    linear_regression(train_x, train_y, pred_x, review_id, get_model=True)
    ridge_regression(train_x, train_y, pred_x, review_id, v_curve=True)
    print "modelling done.."





