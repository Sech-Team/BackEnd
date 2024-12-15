import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import os
import boto3
from dotenv import load_dotenv
from io import StringIO

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
    # data = pd.read_csv("./real_estate.csv", index_col="No")

    # return data

# Ham loc du lieu tach du lieu vao va gia tri muc tieu
def split_data(data):
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    return X, Y

# Ham tach testcases de train va test    
def split_test(X, Y, train_sz, test_sz):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_sz, test_size=test_sz, random_state=0)
    return X_train, X_test, Y_train, Y_test

# Ham tao bang Max
def get_max(data):
    Max_values = data.max()
    return pd.Series(Max_values)

# Ham tao bang Min
def get_min(data):
    Min_values = data.min()
    return pd.Series(Min_values)

# Ham chuan hoa du lieu theo tieu chuan Norm    
def Norm_Feature_Scaling(data):
    # Ham viet tay
    for name_of_col in data.columns:
        Max_value = data[name_of_col].max(axis=0)
        Min_value = data[name_of_col].min(axis=0)
        data[name_of_col] = (data[name_of_col] - Min_value) / (Max_value - Min_value)
    return data

def lib_Norm_Feature_Scaling(data, scaler):
    # Ham thu vien
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)

# Ham khoi phuc gia tri chuan hoa viet tay
def revalues(Scaled_data, Min_values, Max_values):
    for name_of_col in Scaled_data.columns:
        Scaled_data[name_of_col] = Scaled_data[name_of_col] * (Max_values[name_of_col] - Min_values) + Min_values
    return Scaled_data

# Ham khoi phuc gia tri chuan hoa cua thu vien
def lib_revalues(Scaled_data, scaler):
    data = scaler.inverse_transform(Scaled_data)
    return pd.DataFrame(data, columns=Scaled_data.columns)

def distance(point1, point2):
    dis = 0
    for tag in point1.index:
        dis += (point1[tag] - point2[tag]) ** 2
    dis = np.sqrt(dis)
    return dis

def predict_value(test, train, train_label, k):
    list_dist = []
    
    for idx in range(len(train)):
        list_dist.append((distance(train.iloc[idx], test), train_label.iloc[idx]))
    
    list_dist.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in list_dist[:k]]
    predicted_value = np.mean(k_nearest_labels)
    return predicted_value

def predict_values(test, train, train_label, k):
    predictions = []
    for idx in range(len(test)):
        predicted_value = predict_value(test.iloc[idx], train, train_label, k)
        predictions.append(predicted_value)
    
    # Chuyển đổi danh sách dự đoán thành DataFrame
    Y_predict = pd.DataFrame(predictions, index=test.index, columns=["Predicted Values"])
    return Y_predict

def update(new_data, path="real_estate.csv"):
    csv_buffer = StringIO()

    data = read()
    new_data_df = pd.DataFrame(new_data)
    data = pd.concat([data, new_data_df], ignore_index="No")
    data.to_csv(csv_buffer, index_label="No")
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY, Body=csv_buffer.getvalue())
    
def cal_predict(newdata):
    # Doc du lieu
    data_read = read()

    # Phan chia va loc du lieu
    X, Y = split_data(data_read)
    X_train, X_test, Y_train, Y_test = split_test(X, Y, 0.9, 0.1)

    # Tao scale
    X_train_scaler = MinMaxScaler()
    Y_train_scaler = MinMaxScaler()

    X_test_scaler = MinMaxScaler()
    Y_test_scaler = MinMaxScaler()

    # Luu lai Min va Max de revalue (danh cho cac ham viet tay)
    X_train_Max = get_max(X_train)
    X_train_Min = get_min(X_train)

    X_test_Max = get_max(X_test)
    X_test_Min = get_min(X_test)

    Y_train_Max = get_max(Y_train.to_frame())
    Y_train_Min = get_min(Y_train.to_frame())

    Y_test_Max = get_max(Y_test.to_frame())
    Y_test_Min = get_min(Y_test.to_frame())

    X_test = lib_Norm_Feature_Scaling(X_test, X_test_scaler)
    X_train = lib_Norm_Feature_Scaling(X_train, X_train_scaler)
    
    k = 22
    Y_predicted = predict_values(X_test, X_train, Y_train, k)

    newdata_df = pd.DataFrame([newdata], columns=X_train.columns)
    newdata_scaled = lib_Norm_Feature_Scaling(newdata_df, X_train_scaler)
    new = predict_value(newdata_scaled.iloc[0], X_train, Y_train, k)
    newdata.append(new)

    newdata = pd.DataFrame([newdata], columns=data_read.columns)
    update(newdata)
    return new
