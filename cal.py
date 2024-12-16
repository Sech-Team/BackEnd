from sklearn.preprocessing import MinMaxScaler
import os
import boto3
from dotenv import load_dotenv
from io import StringIO
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