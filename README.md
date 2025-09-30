# Stationarity Tests Automation in Python

## Overview
Multiple linear regression can only be applied to **stationary time series**.  
A series is said to be stationary if its **mean**, **variance**, and **autocovariance** are constant over time:

$\mathbb{E}[Y_t] = \mu, \quad Var(Y_t) = \sigma^2, \quad Cov(Y_t, Y_{t-k}) = \gamma(k)$

In R, several functions exist to automate **stationarity testing**, but Python only provides the `adfuller()` function from `statsmodels`. While powerful, it has some limitations:
- It allows the user to specify a constant, a trend, or no deterministic component, and the maximum number of lags.  
- It returns the test statistic, MacKinnon’s approximate p-value, the number of lags, and critical values at standard significance levels.  

However, `adfuller()` does **not provide information on the significance of the regression coefficients** (constant, trend, lag). Moreover, it only tests for the presence of a unit root, without distinguishing whether the series is **trend-stationary (TS)** or **difference-stationary (DS)**.

---

## Contribution of this project
To address these limitations, this project introduces **automated stationarity testing** through custom Python functions:
- **`reg_test`**: evaluates the significance of regression coefficients on the first-differenced series.  
- **`detrend_series`**: removes deterministic trends when necessary.  
- **`stationarity`**: the main function, integrating the above steps to fully automate the stationarity test.  

The final output includes:
- The retained specification (constant, trend, none).  
- The significance of the associated coefficients.  
- The optimal lag order.  
- A conclusion on whether the series is **TS** or **DS**.  
- Graphs of the transformed stationary series and its Partial Autocorrelation Function (PACF).  

---

## How the program works
1. **Initial regression**: The program starts by evaluating the significance of coefficients on the first-differenced series (since the ADF test applies on ΔY).  
2. **Specification loop**: A `for` loop tests successively the three cases:
   - Constant only  
   - Constant + trend  
   - No constant, no trend  
3. **Decision rule**:  
   - If at least one specification has significant coefficients, the ADF test is executed under this specification.  
   - If none are significant:
     - If the series is **DS**, a first differencing is applied:  
       $\Delta Y_t = Y_t - Y_{t-1}$  
     - If the series is **TS**, the deterministic trend is removed:  
       $Y_t^{detrended} = Y_t - \hat{\beta} t$
   - The process is repeated on the transformed series.  

---

## Project structure
stationarity-tests/
├── stationarity_tests.py
├── examples.ipynb
├── requirements.txt
└── README.md

