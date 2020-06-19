#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import itertools
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
# getting cov-19 data from github 
def get_data():
    url='https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    data = pd.read_csv(url)
    df = pd.DataFrame(data)
    df = df.loc[:,['date','state','cases']]
#    df = df.loc['2020-01-31':]
    df['case']=df.groupby('state').diff()
#    df.groupby('state')[0] = df.groupby('state')[1]
    for i in range(len(df['state'])):
        if np.isnan(df['case'][i]):
            df['case'][i] = df['cases'][i]
    df.sort_values(by=['state','date'], inplace=True)
    df.rename(columns={'cases':'cumcase'}, inplace=True)
    df['daily growth'] =np.log(df.cumcase).groupby(df.state).diff()
#    df['daily growth'] = df.groupby('state')['cumcase'].pct_change()
    for i in range(len(df['state'])):
        if np.isnan(df['daily growth'][i]):
            df['daily growth'][i] = 0
    return df 

def get_recover():
    url = 'https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv'
    data = pd.read_csv(url)
    df = pd.DataFrame(data)
    result = df.T.iloc[4:,]
    result.columns = df.T.iloc[1,:]
    return result['US']

    
# plot the live cumulative case of each state
def get_plot(df):
    states = df.state.unique()
    for i in range(len(states)):
        sd = df[(df.state == states[i])]
        cum = sd['cumcase']
        growth = sd['daily growth']
        date = sd['date']
    #    a = pd.DataFrame(date, cum, growth)
    #    x1 = np.linspace(0.0, 5.0)
    #    x2 = np.linspace(0.0, 2.0)
        begin=date.iloc[0]
#        print(begin)
        end = date.iloc[-1]
        midlen = len(date)//2
        
        mid = date.iloc[midlen]
        y1 = cum
        y2 = growth
        
        plt.subplot(2, 1, 1)
        
        plt.plot(date, y1, 'g')
        plt.title('The live data of '+states[i])
        plt.ylabel('cumulative case')
        plt.xticks([0, midlen, len(date)],[begin, mid,end])
        plt.subplot(2, 1, 2)
        plt.plot(date, y2, 'r')
        plt.xlabel('date')
        plt.ylabel('growth rate %')
        plt.xticks([0, midlen, len(date)],[begin, mid,end])
        
        plt.show()
        
# plot the United States live data
def plot_cumulative(df):
    cumu_df = get_data().groupby('date').sum()
    cumu_df['date'] = cumu_df.index
    cumcase = cumu_df['cumcase']
    dailygrowth = cumu_df['daily growth']
    recover = get_recover()
    date = cumu_df['date']
    begin=date.iloc[1]
    end = date.iloc[-1]
    midlen = len(date)//2
    mid = date.iloc[midlen]
    y1 = cumcase
    y2 = dailygrowth
    y3 = recover
    plt.subplot(3, 1, 1)
        
    plt.plot(date, y1, 'g',)
    plt.title('The live data of the United States',fontsize = 15)
    plt.ylabel('cumulative case',fontsize=12)
    plt.xticks([0, midlen, len(date)],[begin, mid,end],fontsize=12)
    plt.subplot(3, 1, 2)
    plt.plot(date, y2, 'b')
#    plt.xlabel('date',fontsize=12)
    plt.ylabel('growth rate %',fontsize=12)
    plt.xticks([0, midlen, len(date)],[begin, mid,end],fontsize=12)
    plt.subplot(3, 1, 3)
    plt.plot(date, y3, 'b')
    plt.xlabel('date',fontsize=12)
    plt.ylabel('rocover %',fontsize=12)
    plt.xticks([0, midlen, len(date)],[begin, mid,end],fontsize=12)

#fit time series data with a seasonal ARIMA model, our first goal is to find the values of ARIMA(p,d,q)(P,D,Q)s that optimize a metric of interest
def test_optimalparamater(y):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    warnings.filterwarnings("ignore")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results = mod.fit()
    
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
#Fitting an ARIMA Time Series Model Producing Forecasts
def get_forecase(y):
    mod = sm.tsa.statespace.SARIMAX(y.astype(float),
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime('2020-05-08'), dynamic=False)
    pred_uc = results.get_forecast(steps=30)
#    pred_ci = pred_uc.conf_int()
    result = pred_uc.predicted_mean
    return result
if __name__ == '__main__':
    df = get_data()
#    states = df.state.unique()
#    plot_cumulative(df)
    cum_df = get_data().groupby('date').sum()
#    forecast = time_seriesforcast(cum_df)
    growth = cum_df['daily growth']
    cum = cum_df['cumcase']
    recover = get_recover()
#    y = cum_df['daily growth']
    for_growth = get_forecase(growth)
    for_cum = get_forecase(cum)
    for_recover = get_forecase(recover)
    report = pd.DataFrame([for_growth,for_cum]).T
    report.columns = ['predicted growth rate','predict cumcase']
    report['predicted recover'] = for_recover.values
    report['predicted infection rate'] = report['predict cumcase']/(328.2*1000000) #current approximate U.S. population 328.2M
    report['recover rate'] = report['predicted recover']/report['predict cumcase']
#    report.to_csv('predict_next30dayscov19data.csv')
#
#    mod = sm.tsa.statespace.SARIMAX(y.astype(float),
#                                order=(1, 1, 1),
#                                seasonal_order=(0, 1, 1, 12),
#                                enforce_stationarity=False,
#                                enforce_invertibility=False)
#    results = mod.fit()
#    print(results.summary().tables[1])
#    pred = results.get_prediction(start=pd.to_datetime('2020-05-01'), dynamic=False)
##    y_forecasted = pred.predicted_mean
##    y_truth = y['2020-05-01':]
##    mse = ((y_forecasted - y_truth) ** 2).mean()
##    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
#    pred_uc = results.get_forecast(steps=30)
#    pred_ci = pred_uc.conf_int()
#    pred_uc.predicted_mean
#ARIMA(1, 1, 1)x(0, 1, 1, 12)12 - AIC:309.9278723312533
#ARIMA(1, 1, 1)x(0, 1, 1, 12)12 - AIC:309.9278723312533
#    forecast = forecast(cum_df)
#    combine = forecast(cum_df)
#    acf = acfpacf(cum_df)
#    test_stationarity(ts)
#    plot = get_plot(df)
#    df.sort_values(by=['date'], inplace=True)
#    for i in range(len(df['state'])):
#        if np.isnan(df['differ'][i]):
#            df['differ'][i] = df['cases'][i]
    mod = sm.tsa.statespace.SARIMAX(growth.astype(float),
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    pred = results.get_prediction(start=pd.to_datetime('2020-05-08'), dynamic=False)
    a = pred.conf_int().mean(axis = 1)