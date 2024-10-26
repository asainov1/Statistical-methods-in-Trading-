# %%
trad = pd.read_csv('/Users/alikhansainov/Desktop/yolo/QA-DS-TASK-DATA-FROM-2020-REDUCED-COLUMNS.csv')
#Исходные данные:
"""""
Дата сет для проведения исследования нужно скачать по ссылке.
Данные представляют из себя цены закрытия минутных баров (свечей) на склеенные  фьючерсные контракты:
E-MINI S&P 500 FUTURES - фьючерс на фондовый индекс SP500 (США)
FTSE CHINA A50 INDEX FUTURES - фьючерс на индекс Шанхайской фондовой биржи (Китай)
10 YEAR TREASURY NOTE FUTURES - фьючерс на 10-ти летние казначейские облигации США

Пример данных:
Описание полей:
+ Timestamp - временная метка в формате "dd-mm-yyyy HH:MM:SS"
+ Close Candle - цена закрытия инструмента
+ Ticker Full Name - полное наименование инструмента


Задание:
Любыми методами необходимо проверить  - кто является лидером, 
кто фолловером, 
т.е. движение какого из инструментов определяют направление движения какого-то другого инструмента.
Эта зависимость может носить краткосрочный или долгосрочный характер, зависимость может менять местами фолловера и лидера,
или если Вы найдёте иные лаговые связи - отразите это в решении. Время на выполнение тестового задания – 5 рабочий дней
с момента предоставления  Вам ссылки.
Ключевое в задаче - именно лаговость между инструментами, не просто некая связь (коррреляция, коинтеграция и т.д.), 
а конкретно лаговость между парами инструментов. Её может и не существовать - и если так - докажите это.
"""""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf, grangercausalitytests
from datetime import datetime

df = pd.read_csv('/Users/alikhansainov/Desktop/yolo/QA-DS-TASK-DATA-FROM-2020-REDUCED-COLUMNS.csv')
 
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

tickers = df['Ticker Full Name'].unique()
df.set_index('Timestamp', inplace=True)

data = {}
for ticker in tickers:
    data[ticker] = df[df['Ticker Full Name'] == ticker]['Close Candle'].resample('T').last()  

sp500 = data['E-MINI S&P 500 FUTURES']
treasury_note = data['10 YEAR TREASURY NOTE FUTURES']

def plot_cross_corr(series1, series2, lag_max=50):
    lags = np.arange(-lag_max, lag_max + 1)
    cross_corr = [series1.corr(series2.shift(lag)) for lag in lags]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cross_corr, label='Cross-correlation')
    plt.axvline(0, color='red', linestyle='--', label='No Lag')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.title('Cross-Correlation between SP500 and Treasury Note Futures')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_cross_corr(sp500, treasury_note)

df_combined = pd.DataFrame({'SP500': sp500, 'Treasury Note': treasury_note}).dropna()

grangercausalitytests(df_combined[['SP500', 'Treasury Note']], maxlag=5)
grangercausalitytests(df_combined[['Treasury Note', 'SP500']], maxlag=5)


# %%
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# Loading and preprocessing DF
df = pd.read_csv('/Users/alikhansainov/Desktop/yolo/QA-DS-TASK-DATA-FROM-2020-REDUCED-COLUMNS.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

tickers = df['Ticker Full Name'].unique() 
df.set_index('Timestamp', inplace=True)

# Function of Granger test for all pairs
def granger_test(df, max_lag=5):
    results = {}
    tickers = df['Ticker Full Name'].unique()
    
    for i, ticker_1 in enumerate(tickers):
        for ticker_2 in tickers[i+1:]:
            # Time series for two tickers 
            series_1 = df[df['Ticker Full Name'] == ticker_1]['Close Candle']
            series_2 = df[df['Ticker Full Name'] == ticker_2]['Close Candle']
            
            # DF for two time series
            combined_df = pd.DataFrame({ticker_1: series_1, ticker_2: series_2}).dropna()
            
            # Granger test 
            test_result_1 = grangercausalitytests(combined_df[[ticker_1, ticker_2]], max_lag)
            test_result_2 = grangercausalitytests(combined_df[[ticker_2, ticker_1]], max_lag)
            
            # Results saved to dict
            results[f'{ticker_1} -> {ticker_2}'] = test_result_1
            results[f'{ticker_2} -> {ticker_1}'] = test_result_2
    
    return results

# Grander test to all tickers
granger_results = granger_test(df, max_lag=5)

# Results
for pair, result in granger_results.items():
    print(f"\nResults for {pair}:")
    for lag, test in result.items():
        f_test_p_value = test[0]['ssr_ftest'][1]
        if f_test_p_value < 0.05: 
            print(f"Lag {lag}: p-value = {f_test_p_value} (Significant)")
        else:
            print(f"Lag {lag}: p-value = {f_test_p_value} (Not Significant)")

# %%
