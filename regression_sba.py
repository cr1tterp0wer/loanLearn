# Classification Report
# TP: True Positive
# TN: True Negative
# FN: False Negative
# FP: False Positive
# @params precision: positive predictive value, TP/(TP+FP)
# @params recall: (TPR) True Positive Rate, percentage of data correctly identified; TP/(TP+FN)
# @params f1-score: Accuracy
# @params support: Frequency a rule appears in the data
# @params accuracy: Percentage of correct predictions, (TP+TN)/(TP+TN+FP+FN)
# @params 0: class_0: Default = 0
# @params 1: class_1: Default = 1

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# List the importance of each feature
def regression_feature_importance(X, xgboost):
    print("Feature Importance")
    for name, importance in sorted(zip(X.columns, xgboost.feature_importances_)):
        print(name, "=", importance)

# LOGISTICAL REGRESSION MODEL
def logistical_regression(df):
    # Use ONE-HOT encoding for data attributes
    df = pd.get_dummies(df)
    print(df.head())

    # Establish target and feature fields
    # X: input into the model
    # Y: Output from the model
    # Make predictions on X_Val
    # Test the predictions against Y_val
    y = df['Default']
    X = df.drop('Default', axis=1)

    # Scale the feature values prior to modeling
    # Overlapping two standard deviations on top of each other
    # StandardScaler():z = (x - u) / s
    # @params u: mean of training samples
    # @params s: standard deviation of traning samples
    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)

    # Init model
    log_reg = LogisticRegression(random_state=2)

    # Train the model and make predictions
    log_reg.fit(X_train, y_train)
    y_logpred = log_reg.predict(X_val)

    # Print the results
    print("Logistical Regression Model")
    print(classification_report(y_val, y_logpred, digits=3))

# XGBoost REGRESSION MODEL
def xgboost_regression(df):
    df = pd.get_dummies(df)
    y = df['Default']
    X = df.drop('Default', axis=1)

    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)
    print(X_scaled)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)

    xgboost = XGBClassifier(random_state=2)

    # Create our Trees
    xgboost.fit(X_train, y_train)
    y_xgbpred = xgboost.predict(X_val)

    # Print the results
    print("XGB Classifier Model")
    print(classification_report(y_val, y_xgbpred, digits=3))

def xgboost_regression_importance(df):
    df = pd.get_dummies(df)
    y = df['Default']
    X = df.drop('Default', axis=1)

    scale = StandardScaler()
    X_scaled = scale.fit_transform(X)
    print(X_scaled)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)

    xgboost = XGBClassifier(random_state=2)

    # Create our Trees
    xgboost.fit(X_train, y_train)
    y_xgbpred = xgboost.predict(X_val)

    # Print the results
    print("XGB Classifier Model")
    print(classification_report(y_val, y_xgbpred, digits=3))
    regression_feature_importance(X, xgboost);
