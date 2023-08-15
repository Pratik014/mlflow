import os 
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # reading the data as df
    df=pd.read_csv(url,sep=";")
    return df

def evaluate(y_true,y_pred,y_pred_prob):
    #mse=mean_squared_error(y_true,y_pred)
    #mae=mean_absolute_error(y_true,y_pred)
    #rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    accuracy=accuracy_score(y_pred,y_true)
    roc_score=roc_auc_score(y_pred_prob,y_true,multi_class='ovr' )
    
    return accuracy,roc_score



def main(n_estimators,max_depth,min_samples_split,max_samples):
    df=get_data()
    train,test=train_test_split(df,random_state=60)
    X_train=train.drop(['quality'],axis=1)
    X_test=test.drop(['quality'],axis=1)

    y_train=train[['quality']]
    y_test=test[['quality']]

    #el=ElasticNet()
    #el.fit(X_train,y_train)
    #y_pred=el.predict(X_test)

    #mse,mae,rmse=evaluate(y_pred,y_test)
    #print(f'mse {mse},mae {mae},rmse {rmse}')
    with mlflow.run_start():
        rnd=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,max_samples=max_samples)
        rnd.fit(X_train,y_train)
        y_pred=rnd.predict(X_test)
        y_pred_prob=rnd.predict_log_proba(X_test)
    
        mlflow.sklearn.log_model(rnd,'model')

        roc_auc_score,accuracy=evaluate(y_pred,y_test,y_pred_prob)
        
        mlflow.log_metric(accuracy)
        
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("min_samples_split",min_samples_split)
        mlflow.log_param("max_samples",max_samples)        
        print(f'accuracy {accuracy}')

if __name__ == '__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--n_estimators','-n',default=100,type=int)
    arg.add_argument('--max_depth','-m',default=5,type=int)
    arg.add_argument('--min_samples_split','-min',default=5,type=int)
    arg.add_argument('--max_samples','-max',default=5.0,type=float)
    parser_agr=arg.parse_args()
    try:
        main(n_estimators=parser_agr.n_estimators,max_depth=parser_agr.max_depth,min_samples_split=parser_agr.min_samples_split,max_samples=parser_agr.max_samples)
    except Exception as e:
        raise e

