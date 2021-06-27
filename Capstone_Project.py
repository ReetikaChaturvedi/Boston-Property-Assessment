import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from pandas import Series, DataFrame


def getData():
    boston = pd.read_csv (
        "https://data.boston.gov/dataset/e02c44d2-3c64-459c-8fe2-e1ce5f38a035/resource/695a8596-5458-442b-a017-7cd72471aade/download/fy19fullpropassess.csv")

    print ("Data Loaded Successfully !")
    print ("\nThe Shape of Boston Data Frame is {shape}".format (shape=boston.shape))
    print ('\nThe Total Columns of Boston Data Frame are\n {col}'.format (col=boston.columns))
    print ('\nSummary of Target Variable(Total Assessed Value) is given below:\n{summ}'.format (
        summ=(boston['AV_TOTAL'] / 1000).describe ()))
    return boston


def filterData(get_df):
    boston = get_df
    print ("Filter Data for Residential Condominium Units")
    boston = boston[boston['LU'] == 'CD']
    boston = boston[boston['PTYPE'] == 102]
    print ("\nThe Shape of Filtered Boston Data Frame is {shape}".format (shape=boston.shape))
    # Filter the columns that are required to answer the business questions
    boston = boston[
        ['ZIPCODE', 'OWN_OCC', 'AV_TOTAL', 'LAND_SF', 'YR_BUILT', 'YR_REMOD', 'GROSS_AREA', 'LIVING_AREA', 'NUM_FLOORS',
         'U_BASE_FLOOR', 'U_NUM_PARK', 'U_CORNER', 'U_ORIENT', 'U_TOT_RMS', 'U_BDRMS', 'U_FULL_BTH', 'U_HALF_BTH',
         'U_HEAT_TYP', 'U_AC', 'U_FPLACE', 'U_INT_FIN', 'U_INT_CND', 'U_VIEW']]
    print ('\nThe Selected Columns of Boston Data Frame are\n {col}'.format (col=boston.columns))
    print ("\nSummary of Boston Data Frame is given below\n {basicInfo}".format (basicInfo=boston.describe ()))
    # Handle Null values
    boston = boston.dropna (subset=['YR_BUILT'])
    boston['LAND_SF'].fillna ((boston['LAND_SF'].mean ()), inplace=True)
    boston['YR_REMOD'].fillna ((boston['YR_REMOD'].min ()), inplace=True)
    print ("\nThe Shape of cleaned Boston Data Frame is {shape}".format (shape=boston.shape))
    return boston

def olsMethod(X, y):
    X = sm.add_constant (X)
    model = sm.OLS (y, X).fit ()
    print ("OLS Model Summary is shown below:{summ}".format (summ=model.summary ()))
    return None


def linearModel(X_train, y_train, X_test, y_test):
    # Now, Create object of Linear Regression and then fit the linear regression model
    regr = LinearRegression ()
    regr.fit (X_train, y_train)
    # Predict the test set result
    y_predicted = regr.predict (X_test).round (2)
    # The coefficients
    print ('Coefficients: \n', regr.coef_)
    # The coefficient of determination: 1 is perfect prediction
    print ('Coefficient of determination: %.2f'
           % r2_score (y_test, y_predicted))
    return None


def lassoReg(X_train, y_train, X_test, y_test):
    print ("Perform regularization on the data using Lasso Regression")
    # Lasso Regression for regularization
    lasso_obj = Lasso (random_state=0)
    lasso_obj.fit (X_train, y_train)
    pred_2 = lasso_obj.predict (X_test)
    # calculating coefficients
    coeff = DataFrame (X_train.columns)
    coeff['Coefficient Estimate'] = Series (lasso_obj.coef_)
    print (coeff)
    print ('\n\nModel performance on Training data after Regularization is {lasTrain}'.format (
        lasTrain=lasso_obj.score (X_train, y_train)))
    print ('\n\nModel performance on Testing data after Regularization is {lasTest}'.format (
        lasTest=lasso_obj.score (X_test, y_test)))
    residual = y_test["Assessed_Value"].values - pred_2
    return None


