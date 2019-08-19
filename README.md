# SVR_class document:

## Author Info: 
* Name: Kent010341 (Nickname using on internet)
* e-mail: kent010341@gmail.com

---
## Parameters:
* kernel: string or callable method, optional (default='rbf')
  * Kernel type, currently support linear, rbf kernel and custom kernel.
  * Custom kernel must satisfy the Mercer's conditions.
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
* kerneled_matrix: 2-D array
  * Training set after kernel transfer.
* alphas: array
  * Coefficients of the support vector.
* b: float
  * Bias of the prediction function.

## Methods:
* get_params(self):
  * Print the parameters set by initially defined.
* fit(self, train_X, train_y):
  * Fit the model according to the given training data.
* predict(self, test_X, test_y=None, y_type=np.array):
  * Perform regression on testing input, calculating MAPE and RMSE value if test_y is given. Type of the output can be change by y_type.

---
## Reference:
* [Understanding Support Vector Machine Regression](https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html?s_tid=mwa_osa_a)
* [Efficient SVM Regression Training with SMO](https://www.researchgate.net/publication/2360418_Efficient_SVM_regression_training_with_SMO)
* [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
* [SVM code](https://github.com/rongxuanhong/MLCodeOnTheBlog/blob/master/SVM%E4%B9%8BPython%E5%AE%9E%E7%8E%B0%E4%BB%A3%E7%A0%81/svm.py)

