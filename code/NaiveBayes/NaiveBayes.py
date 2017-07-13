import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Execute this function once input data has been vectorised.

TestSplit = 60000

class NaiveBayes():
	def value_counts(self,k):
		uniq = set(list(k))
		d = {}
		for u in uniq:
			d[u] = np.sum(k==u)
		return d

	def fit(self,X,y):
		self.X = np.array(X)
		self.y = np.array(y)
		self.m, self.n = self.X.shape
		Xt = self.X.T
		self.count_value = [self.value_counts(k) for k in Xt]  # list of dictionaries for every feature
		rec_where_pos = self.X[self.y == 1].T
		rec_where_neg = self.X[self.y == 0].T
		self.parameter_pos = []
		self.parameter_neg = []
		for index, item in enumerate(self.count_value):
		   self.parameter_pos.append(self.value_counts(rec_where_pos[index]))
		   self.parameter_neg.append(self.value_counts(rec_where_neg[index]))
		self.total_pos = float(sum(self.y==1))
		self.total_neg = float(sum(self.y==0))
		total = self.total_pos + self.total_neg
		self.prior_prob_pos = self.total_pos / total
		self.prior_prob_neg = self.total_neg / total

	def predict(self,X_test):
		X_test = np.array(X_test)
		m, n = X_test.shape
		predictions = np.zeros(m)
		for i, rows in enumerate(X_test):  
		    probXneg = np.zeros(n)
		    probXpos = np.zeros(n)
		    for j, value in enumerate(rows): 
		        n_count = self.parameter_neg[j].get(value,0)
		        p_count = self.parameter_pos[j].get(value,0)
		        probXpos[j] = (p_count  / (self.total_pos))
		        probXneg[j] = (n_count  / (self.total_neg))
		    predictions[i] = np.prod(probXpos)*self.prior_prob_pos
		    #predictions[i] = np.log(self.prior_prob_pos) - np.log(self.prior_prob_neg) + np.sum(np.log(np.prod(probXpos)))  - np.sum(np.log(np.prod(probXneg)))
		p = predictions
		return p

	
	def TestTrainSplit(self,X,y):
		k = np.random.permutation(X.shape[0])
		train_idx, test_idx = k[:TestSplit], k[TestSplit:]
		X_train,X_test = X[train_idx], X[test_idx]
		y_train,y_test = y[train_idx], y[test_idx]
		return X_train,X_test,y_train,y_test

	def PredictionAccuracy(self,y_pred,y_final):
		return np.mean(y_pred == y_final)

	
	def cross_validation(self,obj,X,y,cv=1):
		scores = []
		preds = []
		splitX = np.split(np.array(X),cv,axis=0)
		splitY = np.split(np.array(y),cv,axis=0)
		for u in range(cv):
			splX = splitX[:]
			splY = splitY[:]
			X_te = splX.pop(u)
			X_tr = np.concatenate(splX,axis=0)
			y_te = splY.pop(u)
			y_tr = np.concatenate(splY,axis=0)
			y1 = []
			y2 = []
			y3 = []
			y4 = []
			for i in y_tr:
				if(i ==1):
					y1.append(1)
				else:
					y1.append(0)
			for i in y_tr:
				if(i ==2):
					y2.append(1)
				else:
					y2.append(0)
			for i in y_tr:
				if(i ==3):
					y3.append(1)
				else:
					y3.append(0)
			for i in y_tr:
				if(i ==4):
					y4.append(1)
				else:
					y4.append(0)

			obj.fit(X_tr,y1)
			ya = obj.predict(X_te)
			obj.fit(X_tr,y2)
			yb = obj.predict(X_te)
			obj.fit(X_tr,y3)
			yc = obj.predict(X_te)
			obj.fit(X_tr,y4)
			yd = obj.predict(X_te)
	
			m = len(yd)
			y_predfinal = []

			for z in range(m):
				temp = max(ya[z],yb[z],yc[z],yd[z])
				if(temp == ya[z]):
					y_predfinal.append(1)
				elif(temp == yb[z]):
					y_predfinal.append(2)
				elif(temp == yc[z]):
					y_predfinal.append(3)
				else:
					y_predfinal.append(4)
			preds.extend(y_predfinal)
			score =  np.mean(y_te == y_predfinal)
			print "Iteration ", u, "Accuracy :",score
			scores.append(score)
		return scores,preds

	def mainfunc(self):
		obj = NaiveBayes()

		x_1 = pickle.load(open('x_train_binary.pkl',"rb"))
		y_1 = pickle.load(open('y_train.pkl',"rb"))

		chi2Estimator = SelectKBest(score_func=chi2, k=100)
		x_1_new = chi2Estimator.fit_transform(x_1, y_1)

		x_1_new = x_1_new.toarray()


		X_train, X_test, y_train, y_test = obj.TestTrainSplit(x_1_new,y_1)

		y1 = []
		y2 = []
		y3 = []
		y4 = []

		for i in y_train:
			if(i ==1):
				y1.append(1)
			else:
				y1.append(0)
		for i in y_train:
			if(i ==2):
				y2.append(1)
			else:
				y2.append(0)
		for i in y_train:
			if(i ==3):
				y3.append(1)
			else:
				y3.append(0)
		for i in y_train:
			if(i ==4):
				y4.append(1)
			else:
				y4.append(0)

		i = 1
		obj.fit(X_train,y1)	
		y_pred1 = obj.predict(X_test)

		i = 2
		obj.fit(X_train,y2)
		y_pred2 = obj.predict(X_test)

		i = 3
		obj.fit(X_train,y3)
		y_pred3 = obj.predict(X_test)

		i = 4
		obj.fit(X_train,y4)
		y_pred4 = obj.predict(X_test)

		m = len(y_pred4)
		y_predfinal = []

		for z in range(m):
			temp = max(y_pred1[z],y_pred2[z],y_pred3[z],y_pred4[z])
			if(temp == y_pred1[z]):
				y_predfinal.append(1)
			elif(temp == y_pred2[z]):
				y_predfinal.append(2)
			elif(temp == y_pred3[z]):
				y_predfinal.append(3)
			else:
				y_predfinal.append(4)

		#uncoment and execute to check the cross validation accuracy.

		'''
		print "cv test"
		s,p =  obj.cross_validation(obj,X_train,y_train,2)
		print "Training Accuracy"
		print obj.PredictionAccuracy(p,y_train)
		'''

		print "Testing Accuracy"
		print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0],(y_test != y_predfinal).sum()))
		print obj.PredictionAccuracy(y_test,y_predfinal)



t = NaiveBayes()
t.mainfunc()
