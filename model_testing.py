from funda_scraper import FundaScraper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
BEST_PARAMS = {
        
    "RandomForestRegressor":  {'n_estimators': 120, 'min_samples_split': 8, 'min_samples_leaf': 3, 'max_depth': 14},
    "SVR":  {'kernel': 'linear', 'degree': 3, 'C': 10.0},
    "Ridge":  {'solver': 'lsqr', 'alpha': 10.0},
    "Lasso":  {'max_iter': 1000, 'alpha': 0.1},
    "XGBRegressor":  {'subsample': 0.7000000000000001, 'reg_lambda': 100.0, 'reg_alpha': 0.01, 'n_estimators': 140, 'min_child_weight': 4, 'max_depth': 9, 'learning_rate': 0.1, 'gamma': 0.2, 'colsample_bytree': 0.7000000000000001}
    }

def preproc_data(data_path):
    df = pd.read_csv(data_path,index_col=0)
    df_clean = df[(np.abs(stats.zscore(df["price"])) < 3)]
    df_reg = df_clean.drop(["house_id","city","zip","lat","lng","address","year_built","price_m2","date_list","ym_list","ym_sold","year_sold","date_sold","term_days","year_list","descrip"],axis=1)
    df_reg = df_reg.apply(lambda x: x.astype('category').cat.codes)
    return df_reg
    
def run_test(data_path,verbose = False):
    def log(x):
        if verbose:
            print(x)
    
    df_reg = preproc_data(data_path)
    
    X_train,X_test,y_train,y_test = train_test_split(df_reg.drop("price",axis=1),df_reg["price"],test_size=0.2,random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # reg = LinearRegression().fit(X_train_scaled, y_train)
    X, y = X_train_scaled, y_train
    best_models = [LinearRegression(),
                   RandomForestRegressor(**BEST_PARAMS["RandomForestRegressor"]),
            SVR(**BEST_PARAMS["SVR"]),
            Ridge(**BEST_PARAMS["Ridge"]),
            Lasso(**BEST_PARAMS["Lasso"]),
            XGBRegressor(**BEST_PARAMS["XGBRegressor"])]
    ensemble_regressor = VotingRegressor(estimators=[('ols', best_models[0]), 
                ('rf', best_models[1]), 
                ('svr', best_models[2]),
                ('xgb', best_models[-1])])
    best_models.append(ensemble_regressor)
    r2_scores_cv = []
    rmses_cv = []
    mae_cv = []
    mses_cv =[]

    for model in best_models:
        model.fit(X_train_scaled,y_train)
        y_pred = model.predict(scaler.transform(X_test))
        log(model.__class__.__name__)
        log("R2 score: {:.2f}".format(r2_score(y_test,y_pred)))
        log("RMSE: {:.2f}".format(np.sqrt(mean_squared_error(y_test,y_pred))))
        log("MAE: {:.2f}".format(mean_absolute_error(y_test,y_pred)))
        r2_scores_cv.append(r2_score(y_test,y_pred))
        rmses_cv.append(np.sqrt(mean_squared_error(y_test,y_pred)))
        mae_cv.append(mean_absolute_error(y_test,y_pred))
        mses_cv.append(mean_squared_error(y_test,y_pred))
    return r2_scores_cv,rmses_cv,mae_cv,mses_cv
