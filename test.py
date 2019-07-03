import numpy as np

data_2016 = np.array([[10], [9], [17], [10]])
data_2017 = np.array([[10], [10], [19], [10]])
data_2018 = np.array([[10], [11], [16], [12]])
data_2019 = np.array([[10], [11], [16], [12]])

data_2016 = np.array([[10], [11], [9], [11], [10], [8], [14], [13], [10], [11], [10], [9]])
data_2017 = np.array([[11], [10], [9], [10], [11], [9], [15], [13], [11], [12], [10], [8]])
data_2018 = np.array([[11], [10], [9], [10], [11], [9], [12], [13], [11], [12], [10], [7]])
data_2019 = np.array([[10], [10], [10], [11], [11], [9], [14], [12], [10], [12], [11], [9]])

train_X = data_2016
train_y = data_2017
test_X = data_2017
test_y = data_2018

def sk():
	from sklearn.svm import SVR
	svr_model = SVR(kernel='rbf',gamma=1)
	svr_model = svr_model.fit(train_X, train_y.T[0])
	pred_y = svr_model.predict(test_X)
	print(pred_y)
	print(svr_model.dual_coef_)

def my():
	from SVR_class import SVR
	svr_model = SVR(kernel='rbf',gamma=1, debug=False)
	svr_model.fit(train_X, train_y.T[0])
	pred_y = svr_model.predict(test_X, test_y)
	print(pred_y)
	print(svr_model.alphas)
	print(svr_model.b)
	print(svr_model.MAPE)
	print(svr_model.RMSE)

sk()
my()