import pandas as pd
import tensorflow as tf
import numpy as np
import  sklearn.ensemble as sk
import os
import time
import shutil
def Count_nan_values(DataFrame):
    for column in DataFrame:
        print(column," NUMBER OF NAN VALUES ",DataFrame[column].isnull().sum())

def Convert_Categorical_to_Numeric(PandasSeries,interpolate=False):

    if (interpolate):
        #assert PandasSeries.isnull().values.any()==True,"Given column posses no nan values"
        PandasSeries.interpolate(inplace=True)
        PandasSeries.replace(PandasSeries.unique(),[i for i in range(PandasSeries.unique().shape[0])],inplace=True)
    else:
        assert PandasSeries.isnull().values.any()==False,"Given column posses some nan values, use INTERPOLATE flag"
        PandasSeries.replace(PandasSeries.unique(), [i for i in range(PandasSeries.unique().shape[0])], inplace=True)

    return PandasSeries

with tf.device ("/DML:0"):
    train_x=pd.read_csv("res/dengue_features_train.csv")
    train_y=pd.read_csv("res/dengue_labels_train.csv")
    test_x=pd.read_csv("res/dengue_features_test.csv")
    submission_df=pd.read_csv("res/submission_format.csv")
    train_x["city"]=Convert_Categorical_to_Numeric(train_x["city"])
    #train_x["week_start_date"]=train_x["week_start_date"].apply(lambda x: pd.to_datetime(x).value)
    train_x.interpolate(inplace=True)

    test_x["city"] = Convert_Categorical_to_Numeric(train_x["city"])
    #test_x["week_start_date"] = train_x["week_start_date"].apply(lambda x: pd.to_datetime(x).value)
    test_x.interpolate(inplace=True)

    train_y["city"] = Convert_Categorical_to_Numeric(train_x["city"])

    train_x.set_index(["city","year","weekofyear"])
    test_x.set_index(["city", "year", "weekofyear"])
    train_y.set_index(["city", "year", "weekofyear"])
    #If it happens every year
    train_x.drop(["year","week_start_date"],axis=1,inplace=True)
    test_x.drop(["year", "week_start_date"], axis=1, inplace=True)
    train_y=train_y["total_cases"]
    #Count_nan_values(train_x)
    #Count_nan_values(train_y)
    #print(train_x.dtypes)

    rf = sk.RandomForestRegressor(n_estimators=600)#300 Maybe just too much? only 1.%% improvment
    rf.fit(train_x, train_y)
    submission_df["total_cases"]=rf.predict(test_x).astype("int")
    submission_df.to_csv("submission.csv",index=False)



