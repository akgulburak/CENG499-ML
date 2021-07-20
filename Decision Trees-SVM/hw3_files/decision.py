from graphviz import Digraph
import numpy as np
import dt
from scipy.stats import chi2

gra1 = Digraph('G')
gra1.attr(size='8,5')

gra2 = Digraph('G')
gra2.attr(size='8,5')

gra3 = Digraph('G')
gra3.attr(size='8,5')

gra4 = Digraph('G')
gra4.attr(size='8,5')
#
#gra.node('a', 'Machine Learning Errors')
#
#gra.node('b', 'RMSE')
#
#gra.node('c', 'MAE')
#
#gra.render('Machine.pdf')

tree_index = 0
nodes = []
edges = []


def chi_table_construct():
	p = np.array([0.995, 0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01, 0.005])
	df = np.array(range(1, 30)).reshape(-1, 1)
	table = chi2.isf(p, df)
	return table,p,df

chi_table,chi_p,chi_df = chi_table_construct()

def decide_split(data,labels,gain):
	gains = np.zeros((data.shape))
	maximum = -1
	minimum = -1

	attr_index = -1
	split = -1
	if(gain=="info_gain"):
		for i in range(data.shape[1]):
			a = dt.calculate_split_values(data, labels, 3, i, gain)
			tmp = a[np.argmax(a[:,1])]
			if tmp[1]>maximum:
				maximum=tmp[1]
				attr_index = i
				split = tmp[0]
	elif(gain=="avg_gini_index"):
		a = dt.calculate_split_values(data, labels, 3, 0, gain)
		tmp = a[np.argmin(a[:,1])]
		minimum=tmp[1]
		attr_index = 0
		split = tmp[0]

		for i in range(1,data.shape[1]):
			a = dt.calculate_split_values(data, labels, 3, i, gain)

			tmp = a[np.argmin(a[:,1])]
			if tmp[1]<minimum:
				minimum=tmp[1]
				attr_index = i
				split = tmp[0]

	return split,attr_index

def get_label_arr(labels,num_labels):
	(unique, counts) = np.unique(labels, return_counts=True)
	numbers = np.zeros((num_labels))
	for i in range(len(unique)):
		numbers[unique[i]]=int(counts[i])
	return numbers	

def construct_tree(data,labels,index,num_labels,edge_label,preprune,gain):
	global tree_index, nodes, edges
	tree_index+=1
	(unique, counts) = np.unique(labels, return_counts=True)

	numbers = np.zeros((num_labels))
	for i in range(len(unique)):
		numbers[unique[i]]=int(counts[i])
	if len(unique)<=1:
		string_num = str(np.asarray(numbers,dtype=int).tolist())
		nodes.append([str(tree_index),string_num])
		edges.append([str(index), str(tree_index),edge_label])
		tree_index+=1
		return

	split, attr_index = decide_split(data, labels,gain)
	if tree_index!=1:
		edges.append([str(index), str(tree_index),edge_label])

	left_bucket = data[data[:,attr_index]<split]
	right_bucket = data[data[:,attr_index]>=split]
	left_labels = labels[data[:,attr_index]<split]
	right_labels = labels[data[:,attr_index]>=split]

	left_arr = get_label_arr(left_labels,num_labels)
	right_arr = get_label_arr(right_labels,num_labels)

	parent_index = tree_index
	chi_amount, degree = dt.chi_squared_test(left_arr,right_arr)
	#print(chi_amount,degree,left_arr,right_arr,chi_table[degree][5])
	#print(chi_amount,degree,chi_table[degree-1][4])
	if chi_amount>chi_table[degree-1][5] or preprune==False:
		string_num = str(np.asarray(numbers,dtype=int).tolist())
		nodes.append([str(tree_index), 'x['+str(attr_index)+']<'+str(split)+"\n"+string_num])
		construct_tree(left_bucket,left_labels,parent_index,num_labels,"<",preprune,gain)
		construct_tree(right_bucket,right_labels,parent_index,num_labels,">=",preprune,gain)
	else:
		string_num = str(np.asarray(numbers,dtype=int).tolist())
		nodes.append([str(tree_index),string_num])

	#return str(tree_index),'x['+str(attr_index)+']<'+str(split)+"\n"+str(counts)

def draw_tree(gra):
	global nodes, edges
	nodes.sort(key=lambda x: int(x[0]))

	for i in range(len(nodes)):
		gra.node(str(nodes[i][0]),str(nodes[i][1]))
	for i in range(len(edges)):
		gra.edge(str(edges[i][0]),str(edges[i][1]),str(edges[i][2]))	

	nodes = []
	edges = []
	tree_index = 0

def find_node(node):
	for i in range(len(nodes)):
		if(nodes[i][0]==str(node)):
			return i

def find_edge(node,comparison):
	found = -1
	for i in range(len(edges)):
		if(edges[i][0]==str(node) and edges[i][2] == comparison):
			found = edges[i][1]			
			break
	return found

def find_max(arr):
	maximum=-1
	for i in range(len(arr)):
		if int(arr[i])>maximum:
			maximum=int(arr[i])
			label = i
	return label

def test(x,y):

	correct = 0
	for i in range(x.shape[0]):
		node = find_node(1)
		node_number = 1

		while True:
			find_start = nodes[node][1].find("x[")
			if(find_start==-1):
				labels_ = nodes[node][1][1:-1]

				labels_ = labels_.split(",")

				found_label = find_max(labels_)
				if found_label == y[i]:
					correct+=1

				break

			find_start = find_start+2

			find_end = nodes[node][1].find("]")
			find_enter = nodes[node][1].find("\n")

			attr_index = nodes[node][1][find_start:find_end]
			split_val = nodes[node][1][find_end+2:find_enter]

			attr_index = int(attr_index)
			split_val = float(split_val)

			if(x[i][attr_index]<split_val):
				next_val = find_edge(node_number,"<")
			else:
				next_val = find_edge(node_number,">=")
			node = find_node(next_val)
			node_number = next_val
	
	return correct

def main():
	global tree_index
	train_data = np.load('hw3_data/iris/train_data.npy')
	train_labels = np.load('hw3_data/iris/train_labels.npy')
	test_data = np.load ('hw3_data/iris/test_data.npy')
	test_labels = np.load ('hw3_data/iris/test_labels.npy')
	
	unique = np.unique(train_labels)

	construct_tree(train_data,train_labels,0,len(unique),"=",True,"info_gain")
	test_result = test(test_data, test_labels)
	print("Test accuracy for info gain with prepruning: ", test_result/test_data.shape[0])
	draw_tree(gra1)
	gra1.render('info_gain_preprune.pdf')
	tree_index = 0
	
	construct_tree(train_data,train_labels,0,len(unique),"=",False,"info_gain")
	test_result = test(test_data, test_labels)
	print("Test accuracy for info gain without prepruning: ", test_result/test_data.shape[0])
	draw_tree(gra2)
	gra2.render('info_gain_no_preprune.pdf')
	tree_index = 0
	
	construct_tree(train_data,train_labels,0,len(unique),"=",True,"avg_gini_index")
	test_result = test(test_data, test_labels)
	print("Test accuracy for average gini index with prepruning: ", test_result/test_data.shape[0])
	draw_tree(gra3)
	gra3.render('avg_gini_index_preprune.pdf')
	tree_index = 0
	
	construct_tree(train_data,train_labels,0,len(unique),"=",False,"avg_gini_index")
	test_result = test(test_data, test_labels)
	print("Test accuracy for average gini index without prepruning: ", test_result/test_data.shape[0])
	draw_tree(gra4)
	gra4.render('avg_gini_index_no_preprune.pdf')
	tree_index = 0
	
main()