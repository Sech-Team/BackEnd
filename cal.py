import os
import boto3
from dotenv import load_dotenv
from io import StringIO
import pandas as pd
from proc import *
load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_FILE_KEY = "real_estate.csv"
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Ham doc du lieu vao tu CSV
def read():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY)
        df = obj['Body'].read().decode('utf-8')  
        df = pd.read_csv(StringIO(df), index_col="No")

        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def update(new_data):
    csv_buffer = StringIO()

    data = read()
    new_data_df = pd.DataFrame(new_data)
    data = pd.concat([data, new_data_df], ignore_index="No")
    data.to_csv(csv_buffer, index_label="No")
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY, Body=csv_buffer.getvalue())

def cal_predict(new_data):
    data_read = read()

    X, Y = split_data(data_read)
    X_train, X_test, Y_train, Y_test = split_test(X, Y, 0.9, 0.1)

    X_train, X_test, scaler_X = Norm_Feature_Scaling_Lib(X_train, X_test)

    k = 7
    new_data_df = pd.DataFrame([new_data], columns=X_train.columns)
    new_data_df = replace_missing_with_mean(new_data_df, X)
    new_data_scaled = pd.DataFrame(scaler_X.transform(new_data_df), columns=new_data_df.columns)
    predicted_value = predict_value(new_data_scaled.iloc[0], X_train, Y_train, k)
    new_data_df['Y house price of unit area'] = predicted_value
    print(new_data_df)
    update(new_data_df)
    return predicted_value

def filter(df, values):
    # values[0] and values[1] are the range of the first column
    # values[2] and values[3] are the range of the second column
    # values[4] and values[5] are the range of the third column
    # values[6] and values[7] are the range of the fourth column
    # values[8] and values[9] are the range of the fifth column
    # values[10] and values[11] is the range of the sixth column
    minConstraint = df.min()
    maxConstraint = df.max()
    columns = df.columns.tolist()
    j = 0
    for i in range(0, 12, 2):
        minConstraint[columns[j]] = values[i]
        j = j + 1
    j = 0
    for i in range(1, 12, 2):
        maxConstraint[columns[j]] = values[i]
        j = j + 1

    for col in columns:
        df = df[minConstraint[col] <= df[col]]
        df = df[df[col] <= maxConstraint[col]]
    df = df.reset_index()
    df = df.reset_index(drop=True)

    return df
