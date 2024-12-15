import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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