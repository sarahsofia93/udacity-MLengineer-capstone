from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):

    #x_df = data.to_pandas_dataframe().dropna()
    x_df = data.copy()

    # Drop rows with missing values
    x_df = x_df.dropna()

    # job title, job category, employee residence, company location, employment type, work setting
    # One-hot encode categorical variables
    one_hot_columns = ["job_title", "job_category", "employee_residence", "company_location",  "employment_type", "work_setting"]
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(x_df[one_hot_columns])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names(one_hot_columns))
    x_df = x_df.drop(one_hot_columns, axis=1)
    x_df = pd.concat([x_df, one_hot_encoded_df], axis=1)


    # experience level and company size are ordinal values
    # map ordinal values to integers 'Mid-level', 'Senior', 'Executive', 'Entry-level'
    ordinal_columns = ["experience_level", "company_size"]

    ordinal_encoder = OrdinalEncoder(categories=[["Entry-level", "Mid-level", "Senior", "Executive"], ["S", "M", "L"]])
    ordinal_encoded = ordinal_encoder.fit_transform(x_df[ordinal_columns])
    ordinal_encoded_df = pd.DataFrame(ordinal_encoded, columns=ordinal_columns)
    x_df = x_df.drop(ordinal_columns, axis=1)
    x_df = pd.concat([x_df, ordinal_encoded_df], axis=1)






    unique_years = x_df['work_year'].unique()
    year_dict = {year: i for i, year in enumerate(sorted(unique_years), 1)}
    x_df['work_year'] = x_df['work_year'].map(year_dict)


    # Drop salary column (would give away the answer)
    ## drop currency column (not important for the model)
    x_df = x_df.drop(["salary", "salary_currency"], axis=1)


    y_df = x_df.pop("salary_in_usd")
    
    return x_df, y_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")
    parser.add_argument('--alpha', type=float, default=0.0001, help="Inverse of regularization strength")

    args = parser.parse_args()

    run = Run.get_context()
    run.log("Max iterations:", np.int(args.max_iter))




    #url = "https://github.com/sarahsofia93/udacity-MLengineer-capstone/blob/master/jobs_in_data.csv"

    #ds = TabularDatasetFactory.from_delimited_files(path=url)

    ds = pd.read_csv("./jobs_in_data.csv", header=0, delimiter=",")

    x, y = clean_data(ds)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = SGDClassifier(alpha = args.alpha, max_iter=args.max_iter).fit(x_train, y_train)
    joblib.dump(model, 'model.joblib')
    
    rsquared = model.score(x_test, y_test)
    run.log("Rsquared", np.float(rsquared))





if __name__ == '__main__':
    main()
    


