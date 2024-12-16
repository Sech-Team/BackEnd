from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Hàm lọc dữ liệu tách dữ liệu vào và giá trị mục tiêu
def split_data(data):
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    return X, Y

# Hàm tách testcases để train và test    
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
def Norm_Feature_Scaling(data, Max, Min):
    # Ham viet tay
    for name_of_col in data.columns:
        data[name_of_col] = (data[name_of_col] - Min[name_of_col]) / (Max[name_of_col] - Min[name_of_col])
    return data

# Hàm chuẩn hóa dữ liệu theo thư viện    
def Norm_Feature_Scaling_Lib(X_train, X_test):
    scaler_X = MinMaxScaler()

    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled, scaler_X

# Ham khoi phuc gia tri chuan hoa viet tay
def revalues(Scaled_data, Min_values, Max_values):
    for name_of_col in Scaled_data.columns:
        Scaled_data[name_of_col] = Scaled_data[name_of_col] * (Max_values[name_of_col] - Min_values[name_of_col]) + Min_values[name_of_col]
    return Scaled_data

# Hàm khôi phục giá trị chuẩn hóa của thư viện
def lib_revalues(Scaled_data, scaler):
    data = scaler.inverse_transform(Scaled_data)
    return pd.DataFrame(data, columns=Scaled_data.columns)

# Hàm tính khoảng cách Euclidean
def distance(train, test):
    return np.sqrt(((train - test) ** 2).sum(axis=1))

# Hàm KNN
def predict_value(test, train, train_label, k):
    distances = distance(train, test)
    sorted_indices = np.argsort(distances)
    k_nearest_labels = train_label.iloc[sorted_indices[:k]]
    predicted_value = np.mean(k_nearest_labels)
    return predicted_value

# Hàm xử lý multi test
def predict_values(test, train, train_label, k):
    predictions = []
    for idx in range(len(test)):
        predicted_value = predict_value(test.iloc[idx], train, train_label, k)
        predictions.append(predicted_value)
    Y_predict = pd.DataFrame(predictions, index=test.index, columns=["Predicted Values"])
    return Y_predict


def replace_missing_with_mean(data, base):
    for col in data.columns:
        if col in base.columns and data[col].isnull().sum() > 0:
            mean_value = base[col].mean()
            data[col].fillna(mean_value, inplace=True)
    return data

def judge_knn(X_train, X_test, Y_train, Y_test, k_values):
    results = []
    for k in k_values:
        Y_predict = predict_values(X_test, X_train, Y_train, k)
        mse = mean_squared_error(Y_test, Y_predict)
        accuracy = np.sqrt(mse)
        results.append((k, accuracy))
    return results