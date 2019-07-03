# SVR_class document:

## Author Info: 
* Name: Kent010341 (Nickname using on internet)
* e-mail: kent010341@gmail.com

---
## Parameters:
* kernel: string, optional (default='rbf')
  * Kernel type, currently support linear and rbf kernel.
* C: float, optional (default=1)
  * Penalty parameter C of the error term.
* gamma: float, optional (default='auto')
  * Coefficient of kernel like rbf kernel.
* epsilon: float, optional (default=0.1)
  * Epsilon in epsilon-SVR
* max_iter: int, optional (default=-1)
  * Maximum iterations to solving problem, using -1 for no limit.
* debug: bool, optional (default=False)
  * Print all traininig detail if debug is True.
* random_seed: int, optional (default=0)
  * Seed for numpy.random.seed and random.seed.

## Attributes:
* K: 2-D array
  * Training set after kernel transfer.
* alphas: array
  * Cofficients of the support vector.
* b: float
  * Bias of the prediction function.
* MAPE: float
  * Return mean absolute percentage error if test_y is given while using predict method.
* RMSE: float
  * Return root mean square error if test_y is given while using predict method.

## Methods:
* fit(self, train_X, train_y):
  * Fit the model according to the given training data.
* predict(self, test_X, test_y=None, y_type=np.array):
  * Perform regression on testing input, calculating MAPE and RMSE value if test_y is given. Type of the output can be change by y_type.

---
## Reference:
* [Understanding Support Vector Machine Regression](https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html?s_tid=mwa_osa_a)
* [Efficient SVM Regression Training with SMO](https://www.researchgate.net/publication/2360418_Efficient_SVM_regression_training_with_SMO)
* [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

