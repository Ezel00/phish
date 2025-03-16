
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import processStats as ps

pathEmbedTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\testEMBED.csv"
pathEmbedTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\trainEMBED.csv"
pathTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA4Test.csv"
pathTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA4Train.csv"

train = pd.read_csv(pathEmbedTrain)  #pd.read_csv(pathTrain)
test =  pd.read_csv(pathEmbedTest) #pd.read_csv(pathTest)
'''
train = pd.read_csv(pathTrain)
test =  pd.read_csv(pathTest)'''
XTrain = train.drop(columns="label")
yTrain = train["label"]
XTest =test.drop(columns="label")
yTest = test["label"]


def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(X_train)
    Xtest = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(Xtrain, y_train)
    yP = model.predict(Xtest)
    mse = mean_squared_error(y_test, yP)
    r2 = r2_score(y_test, yP)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return model, scaler

def unseen(model, scaler, X, y):
    X = scaler.transform(X)
    y_unseen_pred = model.predict(X)
    mse_unseen = mean_squared_error(y, y_unseen_pred)
    r2_unseen = r2_score(y, y_unseen_pred)

    print(f" Model Evaluation on Unseen Data:")
    print(f"Mean Squared Error: {mse_unseen:.4f}")
    print(f"R² Score: {r2_unseen:.4f}")

#try with lasso and ridge to see if it improves anything
def lasso(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'alpha': np.logspace(-4, 4, 50)}  # Log-scale for better tuning
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    ridge_cv.fit(X_train, y_train)
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid, scoring='r2', cv=5)
    lasso_cv.fit(X_train, y_train)
    bestR = ridge_cv.best_estimator_
    bestL = lasso_cv.best_estimator_
    ridge_pred = bestR.predict(X_test)
    lasso_pred = bestL.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    lasso_r2 = r2_score(y_test, lasso_pred)
    print(f" Ridge Regression - Best Alpha: {ridge_cv.best_params_['alpha']}")
    print(f" Test MSE: {ridge_mse:.4f}, R² Score: {ridge_r2:.4f}\n")
    print(f"Lasso Regression - Best Alpha: {lasso_cv.best_params_['alpha']}")
    print(f" Test MSE: {lasso_mse:.4f}, R² Score: {lasso_r2:.4f}")

#to use the barcode stats just get the dataframes from ps, such as ps.XTrain, ps.XTest or ps.XTRainEmbed ad ps.XTestEmbed
lasso(XTrain, yTrain)
'''model, scaler = train(XTrain, yTrain)
unseen(model, scaler, XTest, yTest)'''
