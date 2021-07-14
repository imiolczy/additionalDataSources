##First in Terminal: git clone https://github.com/imiolczy/additionalDataSources.git

from joblib.parallel import _verbosity_filter
import pandas as pd
from datetime import  timedelta, datetime
import os
from zipfile import ZipFile
from os import path
import time


#Import Sklearn Modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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
X_train = e10[['previous_e10','e10_fd606c1d-2f43-47df-8934-d1ea98871767','e10_51d4b432-a095-1aa0-e100-80009459e03a','e10_0d206f86-308b-43ae-a157-b7e625bdf61b','e10_5cb08765-da25-4bee-a118-a5710cb9aae5','e10_bf9d3a1f-c8a0-4ec3-93ed-018900da43c1']].dropna()
e10NA = e10.dropna()
model = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(X_train, e10NA.e10, shuffle=False, train_size=0.8)
model.fit(x_train, y_train)
print('R2 score after fiting linear model:')
print(model.score(x_test, y_test))

#Setting up Function for xGBoost Hyperparameter Gridsearch
#Based on: https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

def hyperParameterTuning(X_train, y_train):
    param_tuning = {'nthread':[-1], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.1, 0.05, 0.04, 0.03, 0.02, 0.01], #so called `eta` value
              'max_depth': [3, 5, 7, 8, 9],
              'min_child_weight': [6, 5, 4, 3, 2] ,
              'subsample':  [0.6, 0.7, 0.8, 0.9],
              'colsample_bytree': [0.9, 0.8, 0.7],
              'n_estimators': [500]
              } #parameters from: https://ieeexplore.ieee.org/document/8093500 

    xgb_model = XGBRegressor(verbosity = 0)

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 0)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

def hyperParameterTuning2(X_train, y_train):
    param_tuning = {'nthread':[-1], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.03], #so called `eta` value
              'max_depth': [3, 5, 7, 8, 9],
              'min_child_weight': [6, 5, 4, 3, 2] ,
              'subsample':  [0.6, 0.7, 0.8, 0.9],
              'colsample_bytree': [0.9, 0.8, 0.7],
              'n_estimators': [300, 500, 800, 1000, 1200]
              }

    xgb_model = XGBRegressor(verbosity = 0)

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 0)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

#Start Hyper Parameter Tuning
start = time.time()
best_parameters = hyperParameterTuning2(x_train, y_train)
end = time.time()
runtime = round((end-start)/60,2)
print('The Hyper Parameter Tuning had a runtime of '+str(runtime)+' Minutes')
print('The beste parameters are: '+str(best_parameters))

