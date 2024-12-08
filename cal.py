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

# Ham doc du lieu vao tu CSV
def read():
    data = pd.read_csv("./real_estate.csv", index_col="No")
    return data

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

def update( new_data, path):
    data = pd.read_csv(path, index_col="No")
    new_data_df = pd.DataFrame(new_data)
    data = pd.concat([data, new_data_df], ignore_index="No")
    data.to_csv(path, index_label="No")
    
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

    # print("Max values in X_test:")
    # print(X_test_Max)
    # print("Min values in X_test:")
    # print(X_test_Min)

    # Chuan hoa du lieu
    X_test = lib_Norm_Feature_Scaling(X_test, X_test_scaler)
    X_train = lib_Norm_Feature_Scaling(X_train, X_train_scaler)
    
    k = 22
    Y_predicted = predict_values(X_test, X_train, Y_train, k)
    # Tạo bảng so sánh 
    # comparison = pd.DataFrame({"Y_test": Y_test.values, "Y_predicted": Y_predicted["Predicted Values"].values}, index=Y_test.index)
    # print("Bảng so sánh giữa Y_test và Y_predicted:")
    # print(comparison)
    # r2 = r2_score(Y_test, Y_predicted)
    # print(f"R^2 Score {k}: {r2}")

    # neighbors = np.arange(1, 100)
    # train_accuracy = np.empty(len(neighbors))
    # test_accuracy = np.empty(len(neighbors))
  
    # # Loop over K values 
    # for i, k in enumerate(neighbors): 
    #     knn = KNeighborsRegressor(n_neighbors=k) 
    #     knn.fit(X_train, Y_train) 
        
    #     # Compute training and test data accuracy 
    #     train_accuracy[i] = knn.score(X_train, Y_train) 
    #     test_accuracy[i] = knn.score(X_test, Y_test) 
    
    # # Generate plot 
    # plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy') 
    # plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy') 

    # plt.legend() 
    # plt.xlabel('n_neighbors') 
    # plt.ylabel('Accuracy') 
    # plt.title('KNN Varying number of neighbors')
    # plt.show() 

    #update
    # newdata = [ float(input("Transaction date: ")),
    #     float(input("House age: ")),
    #     float(input("Distance to the nearest MRT station: ")),
    #     float(input("Number of convenience stores: ")),
    #     float(input("Latitude: ")),
    #     float(input("Longitude: "))]
    newdata_df = pd.DataFrame([newdata], columns=X_train.columns)
    newdata_scaled = lib_Norm_Feature_Scaling(newdata_df, X_train_scaler)

    newdata.append(predict_value(newdata_scaled.iloc[0], X_train, Y_train, k))

    newdata = pd.DataFrame([newdata], columns=data_read.columns)
    update(newdata, "./real_estate.csv")
