##First in Terminal: git clone https://github.com/imiolczy/additionalDataSources.git

import pandas as pd
from datetime import  timedelta, datetime
import os
from zipfile import ZipFile
from os import path
import time


#Import Sklearn Modules
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Import xGboost Modules
from xgboost.sklearn import XGBRegressor

os.chdir('additionalDataSources\DataSetsDataModel')

weather = pd.read_csv('weatherTable.csv')
weather['MESS_DATUM'] = pd.to_datetime(weather.MESS_DATUM, infer_datetime_format=True)
weather = weather.rename(columns={'MESS_DATUM':'date'})
weather['good_weather'] = pd.to_numeric(weather['good_weather'], errors='coerce')

oil = pd.read_csv('oilTable.csv')
oil['date'] = pd.to_datetime(oil.date, infer_datetime_format=True)

events = pd.read_csv('eventsTable.csv')
events['date'] = pd.to_datetime(events.date, infer_datetime_format=True)

holidays = pd.read_csv('holidaysTable.csv')
holidays['date'] = pd.to_datetime(holidays.date, infer_datetime_format=True)

exchRates = pd.read_csv('exchangeRatesTable.csv')
exchRates['date'] = pd.to_datetime(exchRates.date, infer_datetime_format=True)


if path.exists("pricesRealTable.csv")==False:
  with ZipFile('priceTable.zip', 'r') as zipObj:
   zipObj.extractall()

pricesReal = pd.read_csv('pricesRealTable.csv')  
pricesCompetitors = pd.read_csv('pricesCompetitorTable.csv') 

pricesReal['date'] = pd.to_datetime(pricesReal.date, infer_datetime_format=True)
pricesCompetitors['date'] = pd.to_datetime(pricesCompetitors.date, infer_datetime_format=True)


#Create Master Dataframe with all hours
start_date = datetime(2014,7,1,00,00)
end_date = datetime(2021,3,31,23,00) 
delta = timedelta(hours=1) # delta we want to generate dates for

current_date = start_date
all_dates = []

while current_date <= end_date:
  all_dates.append(current_date)
  current_date += delta

df = pd.DataFrame(all_dates, columns=['date'])
#print(df.info())

#Merge all Data-Files with Master DF
df = df.merge(weather,how='left', left_on='date', right_on='date')
df = df.merge(events,how='left', left_on='date', right_on='date')
df = df.merge(oil,how='left', left_on='date', right_on='date')
df = df.merge(holidays,how='left', left_on='date', right_on='date')
df = df.merge(exchRates,how='left', left_on='date', right_on='date')
df = df.drop(columns={'Description'})

#Merge Master-DF with Real Prices
dfReal = df.merge(pricesReal,how='left', on='date')
dfReal = dfReal.drop(columns={'dieselchange','e5change','e10change'})

#Merge Master-DF with Competitors Prices
dfComp = df.merge(pricesCompetitors,how='left', on='date')
dfComp = dfComp.drop(columns={'dieselchange','e5change','e10change'})

#Focus on e10 only
e10 = dfReal.drop(columns={'diesel','e5','previous_diesel','previous_e5','diesel_diff','e5_diff','station_uuid'}) #Dropping unrelevant columns
compID = dfComp.drop_duplicates('station_uuid')['station_uuid'].dropna().values #Get IDs of competitors

for id in compID:
  sub = dfComp[dfComp['station_uuid']==id]
  sub = sub[['date','e10']].rename(columns={"e10": "e10_"+id})
  e10 = e10.merge(sub,how='left',on='date')

#Fit linear Model
X_train = e10[['e10','previous_e10','e10_51d4b432-a095-1aa0-e100-80009459e03a','e10_0d206f86-308b-43ae-a157-b7e625bdf61b','e10_bf9d3a1f-c8a0-4ec3-93ed-018900da43c1','OilPriceDay€']].dropna()
e10NA = X_train[['e10']]
X_train = X_train.drop(columns='e10')
model = RandomForestRegressor()

x_train, x_test, y_train, y_test = train_test_split(X_train, e10NA.e10, shuffle=False, train_size=0.89664)
model.fit(x_train, y_train)
print('R2 score after fiting random forest model:')
print(model.score(x_test, y_test))


#Based on: https://www.kaggle.com/felipefiorini/xgboost-hyper-parameter-tuning

def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'bootstrap': [True, False],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    } #parameters from: https://ieeexplore.ieee.org/abstract/document/8966799

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_


def hyperParameterTuningV2(X_train, y_train):
    param_tuning = {'nthread':[-1], #when use hyperthread, xgboost may become slower
              'learning_rate': [0.03, 0.1], #so called `eta` value
              'bootstrap': [True, False],
              'max_depth': [50, 80, 100],
              'max_features': [1, 2, 3],
              'min_samples_leaf': [1, 2, 3],
              'min_samples_split': [4, 6],
              'n_estimators': [50, 100, 500, 1000]
              }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_


#Start Hyper Parameter Tuning
start = time.time()
best_parameters = hyperParameterTuning(x_train, y_train)
end = time.time()
runtime = round((end-start)/60,2)
print('The Hyper Parameter Tuning had a runtime of '+str(runtime)+' Minutes')
print('The beste parameters are: '+str(best_parameters))
