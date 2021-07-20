from graphviz import Digraph
import numpy as np
import dt
gra = Digraph('unix')
gra.attr(rankdir='LR', size='8,5')
#
#gra.node('a', 'Machine Learning Errors')
#
#gra.node('b', 'RMSE')
#
#gra.node('c', 'MAE')
#
#gra.render('Machine.pdf')

tree_index = 0

def decide_split(data,labels):
	gains = np.zeros((data.shape))
	maximum = -1
	attr_index = -1
	split = -1
	for i in range(data.shape[1]):
		a = dt.calculate_split_values(data, labels, 3, i, "info_gain")
		tmp = a[np.argmax(a[:,1])]
		if tmp[1]>maximum:
			maximum=tmp[1]
			attr_index = i
			split = tmp[0]
	return split,attr_index

def construct_tree(data,labels,index):
	global tree_index
	(unique, counts) = np.unique(labels, return_counts=True)
	if len(unique)<=1:
		return str(tree_index),str(counts)

	split, attr_index = decide_split(data, labels)
	
	gra.node(str(tree_index), 'x['+str(attr_index)+']<'+str(split)+"\n"+str(counts))

	if tree_index!=0:
		gra.edge(str(index-1), str(tree_index), constraint='false')

	tree_index+=1
	left_bucket = data[data[:,attr_index]<split]
	right_bucket = data[data[:,attr_index]>=split]
	left_labels = labels[data[:,attr_index]<split]
	right_labels = labels[data[:,attr_index]>=split]

	att1,f1=construct_tree(left_bucket,left_labels,index+1)
	att2,f2=construct_tree(right_bucket,right_labels,index+1)
	print(att1,att2,f1,f2)
	return str(tree_index),'x['+str(attr_index)+']<'+str(split)+"\n"+str(counts)

def main():
	train_data = np.load('hw3_data/iris/train_data.npy')
	train_labels = np.load('hw3_data/iris/train_labels.npy')
	test_data = np.load ('hw3_data/iris/test_data.npy')
	test_labels = np.load ('hw3_data/iris/test_labels.npy')

	construct_tree(train_data,train_labels,0)
	gra.render('Machine.pdf')
main()