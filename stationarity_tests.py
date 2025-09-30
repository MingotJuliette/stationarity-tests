###################################################################
                       ###### Package #####
###################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

###################################################################
                       ###### Parameters #####
###################################################################
max_lag = 10 
pvalue_seuil = 0.05

serie_test = pd.read_csv("C:/Users/mabil/OneDrive/Bureau/My Digital Editions/Manifest/GitHub/ma_lga_12345.csv")
df = pd.DataFrame(serie_test["MA"])
name = "serie_test"

###################################################################
                       ###### Fonctions #####
###################################################################

def stationnary(df, name):
    
    print("--------------------------------------------------------------------------------------------------------")
    for serie in ["brut", "difference"]:
        
        if serie == "difference" :
            print("The serie is difference-stationary (DS) \n  The first differencing is applied (ΔY_t = Y_t - Y_{t-1}).\n")
            df = pd.DataFrame(df.iloc[:, 0].diff(1).dropna())
        else : df = df.copy()
        
        for ad, spec, digit in zip(["ct", "c","n"], ["constant + trend","constant", "no constant, no trend"], [1,0,0]):
            if (reg_test(df, spec, 0)[0][digit] < pvalue_seuil) and (ad == "ct") and (ts.adfuller(df,  regression=ad, autolag="BIC")[1] < pvalue_seuil) : 
                df = pd.DataFrame(detrend_series(df)[0])
                print(f"detrended serie")
                continue            
            elif (reg_test(df, spec, 0)[0][digit] < pvalue_seuil) and (ts.adfuller(df,  regression=ad, autolag="BIC")[1] < pvalue_seuil) : 
                
                 result = ts.adfuller(df, regression=ad, autolag="BIC")
                 print(f"With the specification : {spec}")
                 print(f"The P-value is : {result[1]}. Thus, the series is stationary with an optimal number of lags equal to : {result[2]} \n")
                 print(f"{reg_test(df, spec, result[2])[0]}\n")
                 
                 # enregistrement de la série
                 pd.DataFrame(df).to_pickle(name)
        
                 fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                 # Premier graphique : série
                 axes[0].plot(df)
                 axes[0].set_title("Stationary series”")
                    
                 # Deuxième graphique : PACF
                 plot_pacf(df.dropna(), ax=axes[1], lags=int(len(df)/2 - 1), method="ywm")
                 axes[1].set_title("PACF")
                    
                 plt.tight_layout()
                 plt.show()
                
                 return
                

def reg_test(data, spec, lag):

    #df = data.astype(float)
    df = data.copy()
    df["target"] = df.iloc[:,0].diff(1)
    df["var_dep_shift"] = df.iloc[:,0].shift(1)

    # Lags
    lag_cols = []
    if lag != 0:
        for i in range(1, lag + 1):
            df[f'lag{i}'] = df["target"].shift(i)
            lag_cols.append(f"lag{i}")

    df_model = df.dropna().copy()

    # Trends
    df_model['trend'] = np.arange(1, len(df_model) + 1)
    df_model["trend_quadratique"] = df_model['trend'] ** 2

    fixed_cols = ["var_dep_shift"]
    if spec == "constant + trend":
        fixed_cols = ["trend"] + fixed_cols
    elif spec == "constant + trend quadratique":
        fixed_cols = ["trend_quadratique"] + fixed_cols

    X = df_model[fixed_cols + lag_cols]
    if spec != "no constant, no trend":
        X = sm.add_constant(X)

    y = df_model["target"]
    model = sm.OLS(y, X).fit()

    return model.pvalues, model.params

def detrend_series(data):
    """Detrend the raw series by removing its linear trend."""
    t = np.arange(1, len(data) + 1)
    X = sm.add_constant(t)
    model_trend = sm.OLS(data.iloc[:,0], X).fit()
    trend_estime = model_trend.params[0] + model_trend.params[1] * t
    return data.iloc[:,0] - trend_estime, trend_estime 