from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import draw
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

def second_part():
	train_data = np.load('hw3_data/nonlinsep/train_data.npy')
	train_labels = np.load('hw3_data/nonlinsep/train_labels.npy')

	#clf = SVC(C=1.0, kernel='linear')
	#clf = SVC(C=1.0, kernel='rbf')
	clf = SVC(C=1.0, kernel='poly')
	#clf = SVC(C=1.0, kernel='sigmoid')

	result = clf.fit(train_data,train_labels)

	x1_min = np.min(train_data[:,0]*2)
	x1_max = np.max(train_data[:,0]*2)

	x2_min = np.min(train_data[:,1]*2)
	x2_max = np.max(train_data[:,1]*2)

	draw.draw_svm(clf, train_data, train_labels, x1_min, x1_max, x2_min, x2_max, target=None)

def accuracy(arr1,arr2):
	correct = 0
	for i in range(arr1.shape[0]):
		if(arr1[i]==arr2[i]):
			correct+=1
	return correct/arr1.shape[0]

def train_loop(parameters,x,y,test_x,test_y):
	for i in parameters["C"]:
		for j in parameters["gamma"]:
			clf = SVC(C=i,kernel="poly",gamma=j)
#			print(clf)

			clf.fit(x, y)

			scores = cross_validate(clf,x,y,cv=5)
			result = clf.predict(test_x)
#			print(scores)

			acc = accuracy(result,test_y)
			print(acc)

def third_part():
	train_data = np.load('hw3_data/fashion_mnist/train_data.npy')/256
	train_labels = np.load('hw3_data/fashion_mnist/train_labels.npy')/256
	test_data = np.load('hw3_data/fashion_mnist/test_data.npy')/256
	test_labels = np.load('hw3_data/fashion_mnist/test_labels.npy')/256
	
	#C = [0.01, 0.1, 1, 10, 100]
	#gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

	C = [0.01]
	C = [100]
	gamma = [0.00001]
	gamma = [1,0.00001]

	parameters = {"gamma":gamma,"C":C}
	svc = SVC(kernel="poly")

	#clf = GridSearchCV(svc,parameters,cv=5)
	clf = svc

	n, x, y = train_data.shape
	train_x = train_data.reshape((n,x*y))
	train_y = train_labels.copy()

	train_y[train_y>0]=int(1)
	train_y = train_y.astype(int)

#	clf.fit(train_x, train_y)
	x1_min = np.min(train_x[:,0]*2)
	x1_max = np.max(train_x[:,0]*2)

	x2_min = np.min(train_x[:,1]*2)
	x2_max = np.max(train_x[:,1]*2)

	
	n, x, y = test_data.shape
	test_x = test_data.reshape((n,x*y))

	test_y = test_labels.copy()
	test_y[test_y>0]=int(1)
	test_y = test_y.astype(int)

	train_loop(parameters,train_x,train_y,test_x,test_y)
	#acc = accuracy(result,test_y)
	#print(acc)

	#draw.draw_svm(clf, train_x, train_y, x1_min, x1_max, x2_min, x2_max, target=None)

#second_part()
third_part()