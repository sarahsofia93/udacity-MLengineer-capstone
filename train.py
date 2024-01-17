from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):

    #x_df = data.to_pandas_dataframe().dropna()
    x_df = data.copy()

    y_df = x_df.pop("salary_in_usd")
    
    return x_df, y_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()



    #url = "https://github.com/sarahsofia93/udacity-MLengineer-capstone/blob/master/jobs_in_data.csv"

    #ds = TabularDatasetFactory.from_delimited_files(path=url)

    ds = pd.read_csv("jobs_in_data.csv", header=0, delimiter=",")

    x, y = clean_data(ds)

    print(x.columns)


if __name__ == '__main__':
    main()
    


