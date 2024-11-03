#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_absolute_error, mean_squared_error

# First, let's load the data and outline the pattern and the first lines.
data = pd.read_csv('QA-DS-TASK-DATA-FROM-2020-REDUCED-COLUMNS.csv')

print(data.info())

print(data.head())

# Let's look at statistical information about numeric columns.
print(data.describe())
# Convert the timestamp to datetime format. Then we will sort the data by timestamp.
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')
data = data.sort_values(by='Timestamp')
# Now let's divide the data by instruments. For convenience, we will rename the columns and combine the data by time stamp.
instruments = data['Ticker Full Name'].unique()
data_dict = {instrument: data[data['Ticker Full Name'] == instrument][['Timestamp', 'Close Candle']] for instrument in instruments}

for instrument in instruments:
    data_dict[instrument].rename(columns={'Close Candle': instrument}, inplace=True)

merged_data = data_dict[instruments[0]]
for instrument in instruments[1:]:
    merged_data = pd.merge(merged_data, data_dict[instrument], on='Timestamp', how='outer')
print(merged_data.info())

# Let's remove missing values ​​(synchronization of time series is necessary).
merged_data.dropna(inplace=True)
# Lagged correlation between all pairs of instruments

# First, let's do a preliminary analysis of the lag correlation between all pairs of instruments. This is not the main step of analysis. We do it in order to catch dependencies and trends.
def calculate_lag_correlation(data, col1, col2, max_lag=60):
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0:
            corr = data[col1].corr(data[col2].shift(-lag))
        else:
            corr = data[col1].shift(lag).corr(data[col2])
        correlations.append(corr)
    return lags, correlations


results = {}
for i in range(len(instruments)):
    for j in range(i + 1, len(instruments)):
        col1 = instruments[i]
        col2 = instruments[j]
        lags, correlations = calculate_lag_correlation(merged_data, col1, col2)
        results[(col1, col2)] = (lags, correlations)

        # Let's visualize
        plt.figure(figsize=(10, 5))
        plt.plot(lags, correlations, label=f'Correlation between {col1} and {col2}')
        plt.axvline(0, color='red', linestyle='--', label='Zero lag')
        plt.xlabel('Lag (in minutes)')
        plt.ylabel('Correlation')
        plt.title(f'Lagged correlation between {col1} and {col2}')
        plt.legend()
        plt.grid()
        plt.show()

# COMMENTS
"""The chart of 10 YEAR TREASURY NOTE FUTURES and E-MINI S&P 500 FUTURES shows that the correlation between these instruments increases with the lag from -60 to +60 minutes. As the lag increases, the correlation increases. It can be concluded that 10-year US Treasury bonds have a lagging effect on the S&P 500 index. However, the correlation remains negative, they move in opposite directions, but with a certain time lag.
The chart of 10 YEAR TREASURY NOTE FUTURES and FTSE CHINA A50 INDEX FUTURES shows that the correlation also increases with the lag, but remains weak and negative. Perhaps there is a slight negative relationship with the time lag. But the correlation level is too low to draw confident conclusions about a strong lag dependence.
The E-MINI S&P 500 FUTURES and FTSE CHINA A50 INDEX FUTURES chart shows a decreasing positive correlation as the lag increases from -60 to +60 minutes. The positive correlation decreases as the lag increases. The S&P 500 index may have a direct impact on the FTSE China A50 index in the short term. But as the time lag increases, the strength of the relationship decreases.
For more confident conclusions, let's conduct a more in-depth statistical analysis."""

# Time series stationarity analysis
# Before analyzing the dependencies, we need to make sure that the time series are stationary, since most time series models require stationarity. We will conduct 2 tests: Dickey-Fuller Test (To test the hypothesis of the presence of a unit root) and KPSS Test (To test the hypothesis of the absence of stationarity). If the tests show that the series are not stationary, 
#we will apply differentiation to eliminate the trend and seasonality. Since our sample is large enough, we will limit the tests to a sample of 5,000 observations.


warnings.filterwarnings("ignore")

sample_size = 5000

for column in merged_data.columns[1:]:
    series = merged_data[column].iloc[:sample_size]

    adf_test = adfuller(series, autolag='BIC')
    print(f"\nDickey-Fuller Test Results for {column}:")
    print(f"  ADF Statistic: {adf_test[0]}")
    print(f"  p-value: {adf_test[1]}")
    print("  Critical Values:")
    for key, value in adf_test[4].items():
        print(f"    {key}: {value}")

    kpss_test = kpss(series, regression='c', nlags=15)
    print(f"\nKPSS Test Results for {column}:")
    print(f"  KPSS Statistic: {kpss_test[0]}")
    print(f"  p-value: {kpss_test[1]}")
    print("  Critical Values:")
    for key, value in kpss_test[3].items():
        print(f"    {key}: {value}")

# All three series, except for the possible weak case of E-MINI S&P 500 FUTURES, are not stationary according to the results of the KPSS test and ADF test. For further analysis, we apply data transformation (first difference) to achieve stationarity. Then we again conduct repeated stationarity tests for the transformed series.

differenced_data = merged_data.copy()

for column in merged_data.columns[1:]:
    differenced_data[column] = merged_data[column].diff().dropna()

differenced_data = differenced_data.dropna()

sample_size = 5000

for column in differenced_data.columns[1:]: 
    series = differenced_data[column].iloc[:sample_size] 

    adf_test = adfuller(series, autolag='BIC')
    print(f"\nDickey-Fuller Test Results for transformed {column}:")
    print(f"  ADF Statistic: {adf_test[0]}")
    print(f"  p-value: {adf_test[1]}")
    print("  Critical Values:")
    for key, value in adf_test[4].items():
        print(f"    {key}: {value}")

    kpss_test = kpss(series, regression='c', nlags=15) 
    print(f"\nKPSS Test Results for transformed {column}:")
    print(f"  KPSS Statistic: {kpss_test[0]}")
    print(f"  p-value: {kpss_test[1]}")
    print("  Critical Values:")
    for key, value in kpss_test[3].items():
        print(f"    {key}: {value}")

# All time series became stationary after applying the difference transformation. This is confirmed by the low p-values ​​in the Dickey-Fuller test and the KPSS statistics, which are within acceptable limits. Now these series can be used for further analysis of time dependence, model building and lag analysis.
            
# Building a VAR model
# First, let’s prepare the data for the VAR model (without the “Time Stamp” column) and determine the required number of lags.

model_data = differenced_data.drop(columns=['Timestamp'])
model = VAR(model_data)

lag_selection = model.select_order(maxlags=20)
print("Оптимальное количество лагов:")
print(lag_selection.summary())

""""
According to the results, the optimal number of lags for the VAR model is determined to be 20, since it minimizes all the main information criteria (AIC, BIC, HQIC, FPE).
Now we can use the optimal number of lags to train the model.
"""


optimal_lags = 20
fitted_model = model.fit(maxlags=optimal_lags)
print(fitted_model.summary())

""""
Let us draw conclusions based on the results of the VAR model. Many coefficients have significant t-statistics with p-value < 0.05. This indicates that these lags are significant 
for explaining the current value of the dependent variable. For example, for the equation "10 YEAR TREASURY NOTE FUTURES", the lags of the instrument itself (L1, L2, L3, etc.) and 
the lags of "E-MINI S&P 500 FUTURES" and "FTSE CHINA A50 INDEX FUTURES" affect its dynamics. The sign and magnitude of the coefficients show the direction and strength of the influence. 
For example, negative coefficients demonstrate an inverse relationship, and positive ones - a direct one. Some coefficients are small (this means a weak influence of the corresponding lag). 
The constants in all equations are insignificant (p-value > 0.05). This indicates their weak role in explaining the dynamics. The residuals between different time series have a noticeable correlation.
For example, "E-MINI S&P 500 FUTURES" and "FTSE CHINA A50 INDEX FUTURES" have a positive correlation (0.44). This may indicate that their movements are synchronized.
"""

# Impulse Responses (IRF)
# We will plot the impulse responses (IRF) and shock impulse responses for each instrument separately.
irf = fitted_model.irf(10)

irf.plot(orth=False)
plt.show()

for impulse in model_data.columns:
    irf.plot(impulse=impulse, orth=False)
    plt.title(f'Импульсный отклик на шок в {impulse}')
    plt.show()


# Granger Causality Test
# Next, we will perform a causality test for each pair of instruments with the maximum number of lags. For convenience, we will derive pairs with potential causality (p-value < 0.05).

def granger_causality_analysis(data, max_lag=20):
    results = {}
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i != j:
                test_result = grangercausalitytests(data[[col2, col1]], max_lag, verbose=False)
                p_values = [round(test_result[lag][0]['ssr_ftest'][1], 4) for lag in range(1, max_lag + 1)]
                results[(col1, col2)] = p_values
                print(f"\nТест на причинность Грейнджера: {col1} -> {col2}")
                print(f"P-values: {p_values}")

    return results

granger_results = granger_causality_analysis(model_data, max_lag=20)

print("\nПары с потенциальной причинностью (p-value < 0.05):")
for (col1, col2), p_values in granger_results.items():
    significant_lags = [lag for lag, p_value in enumerate(p_values, start=1) if p_value < 0.05]
    if significant_lags:
        print(f"{col1} -> {col2} при лагах: {significant_lags}")

""""
The Granger causality test results show that all pairs of time series have significant p-values ​​< 0.05 at all lags tested (1-20). 
This suggests that there is a strong causal relationship between each instrument. Importantly, there is bidirectional causality for all pairs of instruments. This means that each instrument can predict changes in the others, and vice versa.

10 YEAR TREASURY NOTE FUTURES может предсказывать движения E-MINI S&P 500 FUTURES и FTSE CHINA A50 INDEX FUTURES, что делает его потенциальным лидером для этих инструментов. E-MINI S&P 500 FUTURES также оказывает значительное влияние на 10 YEAR TREASURY NOTE FUTURES 
и FTSE CHINA A50 INDEX FUTURES, что указывает на его важность как лидера. FTSE CHINA A50 INDEX FUTURES демонстрирует значительное влияние на оба других инструмента, что делает его частью взаимной причинно-следственной связи.

"""

# Model Quality Assessment
# Splitting the data for train-test validation
train_size = int(0.8 * len(model_data))
train_data = model_data[:train_size]
test_data = model_data[train_size:]
forecast_steps = len(test_data)

var_model_train = VAR(train_data).fit(optimal_lags)

forecast_train = var_model_train.forecast(train_data.values[-optimal_lags:], steps=forecast_steps)
forecast_train_df = pd.DataFrame(forecast_train, columns=model_data.columns, index=test_data.index)

# Calculating MAE, MSE, and RMSE for model quality assessment
mae = mean_absolute_error(test_data, forecast_train_df)
mse = mean_squared_error(test_data, forecast_train_df)
rmse = np.sqrt(mse)

print(f"Forecast MAE: {mae}")
print(f"Forecast MSE: {mse}")
print(f"Forecast RMSE: {rmse}")

# Variance Decomposition 
var_model_full = VAR(model_data).fit(optimal_lags)
fevd = var_model_full.fevd(10)
print("Variance Decomposition at 10 steps ahead:")
print(fevd.summary())

fevd.plot()
plt.suptitle("Forecast Error Variance Decomposition")
plt.show()


# %%
""" Work done: 

1. Stationarity Testing and Differencing
The Dickey-Fuller and KPSS tests showed that the original series were non-stationary. After differencing, all time series became stationary, making them suitable 
for time series modeling, such as VAR.

2. VAR Model Analysis
Using the VAR model with an optimal lag of 10, the model effectively captured dependencies among the three instruments
The forecast suggests temporary volatility in each instrument but with a tendency toward stabilization.

3. Impulse Response Analysis
 a. 10-Year Treasury Note Futures may exhibit an inverse response to shocks in the E-Mini S&P 500 Futures, reinforcing the finding of a negative lagged correlation.
 b. E-Mini S&P 500 Futures and FTSE China A50 Futures showed some immediate positive response to each other's shocks, but this effect dissipated quickly, aligning with 
the observation of a short-term positive correlation.

4. Variance Decomposition Analysis 

 a. 10-Year Treasury Note Futures: The forecast error variance decomposition showed that the 10-Year Treasury Note Futures are almost entirely explained by their own shocks across 
all forecast horizons. This indicates a largely independent behavior and suggests that it doesn’t act as a follower to the other instruments.

 b. E-Mini S&P 500 Futures: The forecast error variance for the E-Mini S&P 500 Futures is also primarily explained by its own shocks, with a minor influence from the 10-Year Treasury Note Futures. 
This implies that the S&P 500 Futures operate largely independently but have a slight dependency on the Treasury Note Futures.

 c. FTSE China A50 Index Futures: The variance decomposition shows that the FTSE China A50 Index Futures’ forecast error variance is significantly influenced by shocks in the E-Mini S&P 500 Futures. 
Over time, up to 80% of its forecast error variance can be attributed to the S&P 500 Futures, making the FTSE China A50 Futures heavily reliant on the S&P 500 Futures.


5. Model Quality Assessment
Model Performance: The error metrics (MAE, MSE, RMSE) are fairly low. The model has reasonable predictive accuracy for the dataset.

6. Final Conclusions
---10-Year Treasury Note Futures and E-Mini S&P 500 Futures---

The lagged correlation analysis and variance decomposition suggest that 10-Year Treasury Note Futures primarily influence themselves and only slightly impact the E-Mini S&P 500 Futures.
Although a negative lagged correlation exists, the variance decomposition indicates that the Treasury Futures do not play a significant role as a leading indicator for the S&P 500 Futures.
Thus, while there is an inverse response, it’s not dominant in explaining the S&P 500 Futures’ movements.

---10-Year Treasury Note Futures and FTSE China A50 Index Futures---

The variance decomposition and lagged correlation both confirm that there is minimal to negligible lagged influence between the Treasury Note Futures and the FTSE China A50 Futures. This implies that they
operate independently, with no reliable time-lagged impact on one another.

---E-Mini S&P 500 Futures and FTSE China A50 Index Futures---

The variance decomposition highlights a strong dependence of the FTSE China A50 Index Futures on the E-Mini S&P 500 Futures, where up to 80% of its forecast error variance is explained by S&P 500 shocks. 
This suggests a short-term leader-follower dynamic, where the S&P 500 Futures act as a leading indicator for the FTSE China A50 Index Futures.


6. Summary:
The 10-Year Treasury Note Futures show independent behavior, with their own shocks explaining nearly all of their forecast error variance. They have minimal influence on the other instruments, suggesting they do not play a significant leader-follower role in this context.
The E-Mini S&P 500 Futures also exhibit a high degree of independence but show a slight influence from the Treasury Note Futures. They serve as the primary leader for the FTSE China A50 Index Futures, which shows substantial reliance on shocks from the S&P 500 Futures.
The FTSE China A50 Index Futures act primarily as a follower to the E-Mini S&P 500 Futures, with up to 80% of their forecast error variance explained by S&P 500 movements.

"""



