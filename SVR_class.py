import numpy as np 
import random
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing

class SVR():
	def __init__(self, kernel='rbf', C=1, gamma='auto', epsilon=0.1, max_iter=-1, debug=False, random_seed=0):
		np.random.seed(random_seed)
		random.seed(random_seed)
		self.C = C
		self.kernelVar = (kernel, gamma)
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.E_toler = 0.001
		self._isFit = False
		self.debug = debug

	def fit(self, train_X, train_y):
		# Raise Exceptions
		train_X = np.array(train_X, dtype=np.float64)
		if not isinstance(train_X, np.ndarray):
			raise ValueError('Training X must be numpy.ndarray or list.')

		train_y = np.array(train_y, dtype=np.float64)
		if not isinstance(train_y, np.ndarray):
			raise ValueError('Training y must be numpy.ndarray or list.')

		train_X = self._checkX(train_X, label='train_X')
		if not len(train_X.shape) == 2:
			raise ValueError('Training set X must be 2-D instead of {:}-D'.format(len(train_X.shape)))

		if not len(train_y.shape) == 1:
			raise ValueError('Training set y must be 1-D instead of {:}-D'.format(len(train_y.shape)))

		if not len(train_X) == len(train_y):
			raise ValueError('Training set X must have same length as Training set y.')
		
		self._isFit = True
		self.train_X = train_X
		self.train_y = train_y

		# -----------------------------------------------------------------------------------------------
		# Preprocessing, using standardlized
		#self.scaler = preprocessing.StandardScaler()
		#self.scaler = preprocessing.StandardScaler().fit(self.train_X)
		#self.train_X = self.scaler.fit_transform(self.train_X)

		# -----------------------------------------------------------------------------------------------
		# Initialize
		self.N, self.N_features = self.train_X.shape
		self.isChanged = np.zeros(self.N)
		self.K = np.mat(np.zeros((self.N, self.N)))
		for i in range(self.N):
			self.K[:, i] = self._kernelTrans(train_X, self.train_X[i])
		self._debugPrint('Training data transfer to:')
		self._debugPrint(self.K)
		self.alphas = np.zeros(self.N)
		self.b = 0

		# -----------------------------------------------------------------------------------------------
		# Training with SMO
		iterNum = 1
		alphaPairsChanged = 0

		while (alphaPairsChanged > 0) or (iterNum == 1):
			alphaPairsChanged = 0
			self._debugPrint('=====================================')
			self._debugPrint('iterNum = {:}'.format(iterNum))
			if iterNum == 1:
				for i in range(self.N):
					self._debugPrint('-------------------------------')
					self._debugPrint('i = {:}'.format(i))
					alphaPairsChanged += self._innerL(i)
			else:
				nonBound = np.nonzero((self.alphas>-self.C)*(self.alphas<self.C))[0]
				self._debugPrint('nonBound = ')
				self._debugPrint(nonBound)
				for i in nonBound:
					self._debugPrint('-------------------------------')
					self._debugPrint('i = {:}'.format(i))
					alphaPairsChanged += self._innerL(i)
			iterNum += 1
			if self.max_iter != -1 and iterNum > self.max_iter:
				break

		return self

	def predict(self, test_X, test_y=None, y_type=np.array):
		# Raise Exceptions
		test_X = np.array(test_X, dtype=np.float64)
		if not isinstance(test_X, np.ndarray):
			raise ValueError('Testing X must be numpy.ndarray or list.')

		test_X = self._checkX(test_X, label='train_X')
		if not len(test_X.shape) == 2:
			raise ValueError('Training set X must be 2-D instead of {:}-D'.format(len(train_X.shape)))

		# predict
		if not self._isFit:
			raise Exception('SVR model hasn\'t been trained.')

		# standardlized
		#test_X = self.scaler.transform(test_X)

		pred_y = list(range(self.N))
		for i in range(self.N):
			transX = self._kernelTrans(self.train_X, test_X[i])
			pred_y[i] = (np.dot(self.alphas, transX.T[0]) + self.b)

		# Error
		if not isinstance(test_y, type(None)):
			try:
				if len(test_y) == len(pred_y):
					self.MAPE = self._calMAPE(test_y, pred_y)
					self.RMSE = self._calRMSE(test_y, pred_y)
			except:
				pass

		return y_type(pred_y) 

	def _checkX(self, X, label):	
		if len(X.shape) == 1:
			print('Warning: {:} is 1-D, automatically reshape to ({:}, {:})'\
				.format(label, X.shape[0], 1))
			return X.reshape(-1, 1)
		else:
			return X

	def _kernelTrans(self, X, sampleX):
		m = len(X)
		K = np.zeros((m, 1))
		if self.kernelVar[0] == 'linear':
			K = X * sampleX.T
		elif self.kernelVar[0] == 'rbf':
			sigma = self.kernelVar[1]
			if sigma == 'auto':
				sigma = np.true_divide(1, self.N_features)
			elif sigma == 'scale':
				sigma = np.true_divide(self.N_features, X.var())
			if sigma == 0: 
				sigma = 1
			for i in range(m):
				deltaRow = (X[i, :] - sampleX)[0]
				rbf_value = np.exp(np.true_divide(deltaRow * deltaRow.T, (-2.0 * sigma ** 2)))
				if rbf_value == np.inf:
					print('Warning: Overflow encountered while using kernel, consider normalize or standardlize X.')
				K[i] = rbf_value
		else:
			raise NameError('Not support kernel type! You can use linear or rbf!')
		return K

	def _innerL(self, i):
		self._debugPrint('alphas = ')
		self._debugPrint(self.alphas)
		self._debugPrint('b =')
		self._debugPrint(self.b)

		Ei = self._calError(i)
		self._debugPrint('Ei =')
		self._debugPrint(Ei)

		isVioletKKT = not self._checkKKT(i, Ei)
		j ,Ej = self._selectJ(i, Ei, isVioletKKT)
		
		if j == -1:
			self._debugPrint('all KKT pass')
			return 0

		self._debugPrint('j = {:}'.format(j))
		self._debugPrint('Ej = {:}'.format(Ej))

		alphaIold = self.alphas[i].copy()
		alphaJold = self.alphas[j].copy()

		L = max(-self.C, alphaIold + alphaJold - self.C)
		self._debugPrint('L =')
		self._debugPrint(L)
		H = min(self.C, alphaIold + alphaJold + self.C)
		self._debugPrint('H =')
		self._debugPrint(H)

		if L == H:
			return 0

		eta = self.K[i, i] + self.K[j, j] - 2.0 * self.K[i, j]
		self._debugPrint('eta =')
		self._debugPrint(eta)
		if eta <= 0:
			self._debugPrint('eta<=0')
			return 0

		# update j
		is_jUpdate = False
		I = alphaIold + alphaJold
		for sgn in [-2, 0, 2, -1, 1]:
			self._debugPrint('try sgn = {:}'.format(sgn))
			tmpJ = alphaJold + np.true_divide((Ei - Ej + self.epsilon * sgn), eta)
			self._debugPrint('tmpJ = {:}'.format(tmpJ))
			if np.sign(I - tmpJ) - np.sign(tmpJ) == sgn:
				self.alphas[j] = tmpJ
				is_jUpdate = True
				break
		if not is_jUpdate:
			return 0
		self._debugPrint('A_j,new = {:}'.format(self.alphas[j]))
		if self.alphas[j] > H:
			self.alphas[j] = H
		if self.alphas[j] < L:
			self.alphas[j] = L
		self._debugPrint('A_j,new,cli = {:}'.format(self.alphas[j]))

		if abs(self.alphas[j] - alphaJold) < 0.00001:
			self._debugPrint('alphas[j] changes too small.')
			self.isChanged[j] = 1
			return 0

		# update i
		self.alphas[i] += (alphaJold - self.alphas[j])
		self._debugPrint('A_i,new = {:}'.format(self.alphas[i]))
		bi = -(Ei + (self.alphas[i] - alphaIold) * self.K[i, i]\
			+ (self.alphas[j] - alphaJold) * self.K[i, j]) + self.b
		bj = -(Ej + (self.alphas[i] - alphaIold) * self.K[i, j]\
			+ (self.alphas[j] - alphaJold) * self.K[j, j]) + self.b
		if self.alphas[i] > -self.C and self.alphas[i] < self.C:
			self.b = bi
			self._debugPrint('bi is avaliable')
		elif self.alphas[j] > -self.C and self.alphas[j] < self.C:
			self.b = bj
			self._debugPrint('bj is avaliable')
		else:
			self.b = np.true_divide((bi + bj), 2)
			self._debugPrint('b update to (bi+bj)/2')
		self.isChanged[i] = 1
		self.isChanged[j] = 1

		return 1
			
	def _calError(self, i):
		s = 0
		for x, y in zip(self.alphas, self.K[:, i].A.T[0]):
			s += x*y
		Ei = float(s) + float(self.b) - self.train_y[i]
		return Ei

	def _checkKKT(self, i, Ei=None):
		if Ei == None:
			Ei = self._calError(i)

		KKTpass = False
		if self.alphas[i] == 0 and abs(Ei) < self.epsilon + self.E_toler:
			KKTpass = True
		if self.alphas[i] != 0 and self.alphas[i] > -self.C and self.alphas[i] < self.C and \
			abs(Ei) < self.epsilon + self.E_toler and abs(Ei) > self.epsilon - self.E_toler:
			KKTpass = True
		if abs(self.alphas[i]) == self.C and abs(Ei) > self.epsilon - self.E_toler:
			KKTpass = True 

		return KKTpass

	def _selectJ(self, i, Ei, isVioletKKT):
		# i and j should be at least one violate KKT conditions.
		seq = []
		E = np.zeros(self.N)-9999
		if not isVioletKKT:
			# Finding i violet KKT conditions.
			for n in range(self.N):
				if n == i:
					continue
				E[n] = self._calError(n)
				if (not self._checkKKT(n, E[n])) and \
					(self.train_X[i] != self.train_X[n] or \
					self.train_y[i] != self.train_y[n]):
					seq.append(n)
		else:
			for n in range(self.N):
				if n == i:
					continue
				if self.train_X[i] != self.train_X[n] or \
					self.train_y[i] != self.train_y[n]:
					seq.append(n)

		self._debugPrint('Avaliable index for selecting j: ')
		self._debugPrint(seq)
		if len(seq) == 0:
			return -1, None
		'''
		maxStep = -1
		for k in seq:
			if E[k] == -9999:
				E[k] = self._calError(k)
			self._debugPrint('E{:} = {:}'.format(k, E[k]))
			step = abs(E[k] - Ei)
			if maxStep < step:
				maxStep = step
				j = k
		'''
		j = np.random.choice(seq)
		if E[j] == -9999:
			E[j] = self._calError(j)

		return j ,E[j]
		
	def _calMAPE(self, arrReal, arrPredict):
		sumValue = 0
		for i in range(len(arrReal)):
			sumValue += abs(np.true_divide((arrReal[i] - arrPredict[i]), arrPredict[i]))
			#sumValue += ((arrReal[i] - arrPredict[i]))

		return np.true_divide(sumValue, len(arrPredict)) * 100

	def _calRMSE(self, arrReal, arrPredict):
		return MSE(arrReal, arrPredict)**0.5

	def _debugPrint(self, string):
		if self.debug:
			print(string)
