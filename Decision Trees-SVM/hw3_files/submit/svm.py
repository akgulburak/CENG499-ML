from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import draw
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
def accuracy(arr1,arr2):
	correct = 0
	for i in range(arr1.shape[0]):
		if(arr1[i]==arr2[i]):
			correct+=1

	return correct/arr1.shape[0]

def train_loop(parameters,x,y,test_x,test_y,kernel):
	for i in parameters["C"]:
		if kernel=="linear":
			clf = SVC(C=i,kernel=kernel)
			scores = cross_validate(clf,x,y,cv=5)
			print("C: ",i," acc: ", np.average(scores["test_score"]))

		else:			
			for j in parameters["gamma"]:
				clf = SVC(C=i,kernel=kernel,gamma=j)
				#print(clf)

				scores = cross_validate(clf,x,y,cv=5)
				#result = clf.predict(test_x)
				#acc = accuracy(result,test_y)
				#print(scores)

				print("C: ",i," gamma: ",j, " acc: ", np.average(scores["test_score"]))

def first_part():
	train_data = np.load('hw3_data/linsep/train_data.npy')
	train_labels = np.load('hw3_data/linsep/train_labels.npy')

	x1_min = np.min(train_data[:,0]*2)
	x1_max = np.max(train_data[:,0]*2)

	x2_min = np.min(train_data[:,1]*2)
	x2_max = np.max(train_data[:,1]*2)

	C = [0.01, 0.1, 1 ,10 , 100]	
	for i in C:
		clf = SVC(C=i,kernel="linear")
		clf.fit(train_data, train_labels)
		draw.draw_svm(clf, train_data, train_labels, x1_min, x1_max, x2_min, x2_max, target=None)

def second_part():
	train_data = np.load('hw3_data/nonlinsep/train_data.npy')
	train_labels = np.load('hw3_data/nonlinsep/train_labels.npy')

	x1_min = np.min(train_data[:,0]*2)
	x1_max = np.max(train_data[:,0]*2)

	x2_min = np.min(train_data[:,1]*2)
	x2_max = np.max(train_data[:,1]*2)

	kernel = ["linear","rbf","sigmoid","poly"]
	#clf = SVC(C=1.0, kernel='linear')
	#clf = SVC(C=1.0, kernel='rbf')
	#clf = SVC(C=1.0, kernel='sigmoid')

	for i in kernel:
		clf = SVC(C=1.0, kernel=i)
		result = clf.fit(train_data,train_labels)
		draw.draw_svm(clf, train_data, train_labels, x1_min, x1_max, x2_min, x2_max, target=None)

def shape_data(data,label):
	n, x, y = data.shape
	data_x = data.reshape((n,x*y))

	data_y = label.copy()
	data_y[label>0]=int(1)
	data_y = data_y.astype(int)

	return data_x, data_y

def third_part_test(C, gamma, kernel):
	train_data = np.load('hw3_data/fashion_mnist/train_data.npy')/256
	train_labels = np.load('hw3_data/fashion_mnist/train_labels.npy')/256
	test_data = np.load('hw3_data/fashion_mnist/test_data.npy')/256
	test_labels = np.load('hw3_data/fashion_mnist/test_labels.npy')/256

	#shape the data
	train_x, train_y = shape_data(train_data, train_labels)
	test_x, test_y = shape_data(test_data, test_labels)

	best_C = C
	best_gamma = gamma
	best_kernel = kernel

	clf = SVC(C=best_C,kernel=best_kernel,gamma=best_gamma)	
	clf.fit(train_x, train_y)

	result = clf.predict(test_x)

	print("Test accuracy is : ",accuracy(result, test_y))



def third_part():
	train_data = np.load('hw3_data/fashion_mnist/train_data.npy')/256
	train_labels = np.load('hw3_data/fashion_mnist/train_labels.npy')/256
	
	#C = [0.01, 0.1, 1, 10, 100]
	#gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

	#C = [100]
	C = [0.01, 0.1, 1, 10, 100]
	#gamma = [1]
	gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

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
#	x1_min = np.min(train_x[:,0]*2)
#	x1_max = np.max(train_x[:,0]*2)
#
#	x2_min = np.min(train_x[:,1]*2)
#	x2_max = np.max(train_x[:,1]*2)

	#train_loop(parameters,train_x,train_y,test_x,test_y)
	kernel = "linear"
	print("Results for linear kernel: ")
	train_loop(parameters,train_x,train_y,train_x,train_y,kernel)

	print("-----------")

	kernel = "rbf"
	print("Results for rbf kernel: ")
	train_loop(parameters,train_x,train_y,train_x,train_y,kernel)

	print("-----------")

	kernel = "poly"
	print("Results for polynomial kernel: ")
	train_loop(parameters,train_x,train_y,train_x,train_y,kernel)

	print("-----------")

	kernel = "sigmoid"
	print("Results for sigmoid kernel: ")
	train_loop(parameters,train_x,train_y,train_x,train_y,kernel)

def oversample(train_x,train_y):
	num_0 = len(train_y[train_y==0])
	num_1 = len(train_y[train_y==1])

	repeat_num = num_1//num_0
	new_x = np.repeat(train_x[train_y==0],repeat_num-1,axis=0)

	appended_x = np.append(train_x,new_x,axis=0)

	new_y = np.repeat(train_y[train_y==0],repeat_num-1,axis=0)
	appended_y = np.append(train_y,new_y)

	return appended_x, appended_y

def undersample(train_x,train_y):
	num_0 = len(train_y[train_y==0])

	array0_x = train_x[train_y==0]
	array1_x = train_x[train_y==1][:num_0+1]
	appended_x = np.append(array0_x,array1_x,axis=0)

	array0_y = train_y[train_y==0]
	array1_y = train_y[train_y==1][:num_0+1]
	appended_y = np.append(array0_y,array1_y,axis=0)

	return appended_x, appended_y

def fourth_part():
	train_data = np.load('hw3_data/fashion_mnist_imba/train_data.npy')/256
	train_labels = np.load('hw3_data/fashion_mnist_imba/train_labels.npy')/256
	test_data = np.load('hw3_data/fashion_mnist_imba/test_data.npy')/256
	test_labels = np.load('hw3_data/fashion_mnist_imba/test_labels.npy')/256	

	train_x, train_y = shape_data(train_data, train_labels)
	test_x, test_y = shape_data(test_data, test_labels)

	print("Number of 0 labels in test: ", len(test_y[test_y==0]))
	print("Number of 1 labels in test: ", len(test_y[test_y==1]))

	# Direct training code
	print("###################")
	print("For first training:")
	clf = SVC(C=1,kernel="rbf")
	clf.fit(train_x, train_y)
	results = clf.predict(test_x)
	print("Test accuracy is: ", accuracy(results, test_y))
	print("Confusion matrix is: ")
	print(confusion_matrix(test_y, results))

	print("Number of 0 labels in train: ", len(train_y[train_y==0]))
	print("Number of 1 labels in train: ", len(train_y[train_y==1]))

	# Oversampling code
	print("###################")
	print("For Second training:")
	oversample_x, oversample_y = oversample(train_x,train_y)
	clf = SVC(C=1,kernel="rbf")
	clf.fit(oversample_x,oversample_y)
	results = clf.predict(test_x)
	print("Test accuracy is: ", accuracy(results, test_y))
	print("Number of 0 labels in train: ", len(oversample_y[oversample_y==0]))
	print("Number of 1 labels in train: ", len(oversample_y[oversample_y==1]))
	print("Confusion matrix is: ")
	print(confusion_matrix(test_y, results))
	
	# Undersampling code
	print("###################")
	print("For third training:")
	undersample_x, undersample_y = undersample(train_x,train_y)
	clf = SVC(C=1,kernel="rbf")
	clf.fit(undersample_x, undersample_y)
	results = clf.predict(test_x)
	print("Test accuracy is: ", accuracy(results, test_y))
	print("Number of 0 labels in train: ", len(undersample_y[undersample_y==0]))
	print("Number of 1 labels in train: ", len(undersample_y[undersample_y==1]))
	print("Confusion matrix is: ")
	print(confusion_matrix(test_y, results))

	print("###################")
	print("For fourth training:")
	clf = SVC(C=1,kernel="rbf",class_weight="balanced")
	clf.fit(train_x, train_y)
	results = clf.predict(test_x)
	print("Test accuracy is: ", accuracy(results, test_y))
	print("Number of 0 labels in train: ", len(train_y[train_y==0]))
	print("Number of 1 labels in train: ", len(train_y[train_y==1]))
	print("Confusion matrix is: ")
	print(confusion_matrix(test_y, results))
	print("###################")


#############################################
#Uncomment for each part (or together), I didn't uncommented the draw.draw
#functions, so the first and second parts draws and shows the figures on screen
###

#first_part()
#second_part()

#third_part()
#third_part_test(100,0.01,"rbf")

#fourth_part()

###
#############################################