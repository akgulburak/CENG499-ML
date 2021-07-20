import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math as m
import os 

def init_normal(m):
	if type(m) == nn.Linear:
	    #nn.init.uniform_(m.weight)
		nn.init.normal_(m.weight,mean=0,std=0.01)
class MyDataset(Dataset):
	def __init__(self,path,portion):
		self.path = path
		self.portion = portion

		datapath = os.path.join(path,portion)
		self.datas = []

		transformation = transforms.Compose([
			transforms.ToTensor(),
			#transforms.Scale((20,20)),
			transforms.Normalize((0.5,),(0.5,))
			])

		with open(os.path.join(datapath,"labels.txt"),"r") as file:
			lines = file.readlines()
			for i in lines:
				if portion=="train":
					splitted = i.split()
					image = Image.open((os.path.join(datapath,splitted[0])))					
					image = transformation(image)
					self.datas.append(([image,splitted[0]],splitted[1]))
				else:
					splitted = i.split()
					image = Image.open((os.path.join(datapath,splitted[0])))
					image = transformation(image)
					self.datas.append((image,splitted[0]))
	def __len__(self):
		return len(self.datas)

	def __getitem__(self,index):
		if self.portion=="test":
			return self.datas[index][0],self.datas[index][1]

		label = int(self.datas[index][1])
		return self.datas[index][0],label

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel,self).__init__()
		self.fc1 = nn.Linear(4800,10)
		self.relu1 = nn.ReLU()
	def forward(self,x):
		x = x.view(x.size(0),-1)
		x = self.fc1(x)
		x = torch.log_softmax(x,dim=1)
		return x

class MyModel2(nn.Module):
	def __init__(self,hdsize):
		super(MyModel2,self).__init__()
		self.fc1 = nn.Linear(4800,hdsize)
		self.fc2 = nn.Linear(hdsize,10)
	def forward(self,x,activation):
		x = x.view(x.size(0),-1)
		x = self.fc1(x)
		if(activation=="relu"):
			x = F.relu(x)			
		elif(activation=="tanh"):
			x = F.tanh(x)			
		elif(activation=="sigmoid"):
			x = F.sigmoid(x)			
		x = self.fc2(x)
		x = torch.log_softmax(x,dim=1)
		return x

class MyModel3(nn.Module):
	def __init__(self,hdsize):
		super(MyModel3,self).__init__()
		self.fc1 = nn.Linear(4800,hdsize)
		self.fc2 = nn.Linear(hdsize,hdsize)
		self.fc3 = nn.Linear(hdsize,10)
	def forward(self,x,activation):
		x = x.view(x.size(0),-1)
		x = self.fc1(x)
		if(activation=="relu"):
			x = F.relu(x)			
		elif(activation=="tanh"):
			x = F.tanh(x)			
		elif(activation=="sigmoid"):
			x = F.sigmoid(x)			
		x = self.fc2(x)
		if(activation=="relu"):
			x = F.relu(x)			
		elif(activation=="tanh"):
			x = F.tanh(x)			
		elif(activation=="sigmoid"):
			x = F.sigmoid(x)			
		x = self.fc3(x)
		x = torch.log_softmax(x,dim=1)
		return x

def getCorrect(preds,targets):
	correct=0
	for i in range(0,len(targets)):
		if torch.argmax(preds[i])==targets[i]:
			correct+=1
	return correct

def get_valacc(model,dataloader,batch_size,device,activation):
	model.eval()
	running_loss = 0.0
	correct = 0
	for inps, targets in dataloader:
		inps = inps[0]
		inps = inps.to(device)	
		targets = targets.to(device)
		if(activation==None):
			output = model(inps)
		else:
			output = model(inps,activation)

		loss = F.nll_loss(output, targets)
		
		running_loss += loss.item()

		correct += getCorrect(output,targets)

	model.train()
	return (correct/(len(dataloader)*batch_size)),(running_loss/(len(dataloader)))

def load_model(path):
	model = MyModel()
	model.load_state_dict(torch.load(path))
	model.eval()
	return model

def train(model,train_dataloader,validation_dataloader,epochs,optimizer,device,batch_size,expNumber,activation,modelname):
	model = model.to(device)
	maxval = -1
	for i in range(0,epochs):
		count = 0
		correct = 0
		running_loss = 0.0
		acc=0
		for inps, targets in train_dataloader:	
			inps = inps[0]
			inps = inps.to(device)
			targets = targets.to(device)

			optimizer.zero_grad()
			if(activation==None):
				output = model(inps)
			else:
				output = model(inps,activation)
			loss = F.nll_loss(output, targets)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			correct += getCorrect(output,targets)
			count+=1
			acc = correct/(count*batch_size)
			if count%100==0:
				print('\r' + f"Loss: {running_loss/(count):.3f} Accuracy: "+f"{acc:.3f}"+ " " +str(count)+"/"+str(len(train_dataloader)), end='')
			#if count==1:
			#	file = open("predictions.txt","w")
			#	print(running_loss,file=file)
			#	file.close()
		if(activation==None):
			valacc,valloss = get_valacc(model,validation_dataloader,batch_size,device,None) 
		else:
			valacc,valloss = get_valacc(model,validation_dataloader,batch_size,device,activation) 			
		if valacc>maxval:
			maxval=valacc
			if modelname==None:
				torch.save(model.state_dict(), "saved_model_"+str(expNumber))
			else:
				torch.save(model.state_dict(), modelname)
		#valfile = open("val"+str(expNumber)+".txt","a")
		#print("Epoch:"+ str(i) +" valacc:" + str(valacc)+" trainacc:" + str(acc)+", trainaccloss:"+ str(running_loss/count)+ ", valloss:" + str(valloss) ,file=valfile)
		#valfile.close()
		print(" Validation accuracy: ",f"{valacc:.3f}")

def output_labels(path,portion):
	batch_size = 32
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	checkpoint = torch.load("saved_model_64")
	model = MyModel3(2048)
	model.load_state_dict(checkpoint)

	model.eval()
	model.to(device)

	dataset = MyDataset(path,portion)

	dataloader = DataLoader(dataset, batch_size = batch_size)
	file = open("labels.txt","w")
	for inps,paths in dataloader:	
		inps = inps.to(device)

		output = model(inps,"relu")
		for i in range(0,len(inps)):
			print(str(paths[i])+" "+str(torch.argmax(output[i]).item()),file=file)	

def predict(modelpath,path,portion,layersize,hiddensize,activation):
	batch_size = 32
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint = torch.load(modelpath)
	
	if layersize==1:
		model = MyModel()
	elif layersize==2:
		model = MyModel2(hiddensize)
	elif layersize==3:
		model = MyModel3(hiddensize)
	else:
		print("Invalid layer size")

	model.load_state_dict(checkpoint)
	model.to(device)

	dataset = MyDataset(path,portion)

	dataloader = DataLoader(dataset, batch_size = batch_size)
	correct = 0
	if portion=="train":
		file = open("labels.txt","w")
		for inps, targets in dataloader:		
			paths = inps[1]
			inps = inps[0]

			inps = inps.to(device)
			targets = targets.to(device)
			if activation!=None:
				output = model(inps,activation)
			else:
				output = model(inps)

			correct += getCorrect(output,targets)

			for i in range(0,len(inps)):
				print(str(paths[i])+" "+str(torch.argmax(output[i]).item()),file=file)	

		print(correct/(len(dataloader)*batch_size))
	else:
		file = open("labels.txt","w")
		for inps,paths in dataloader:	
			inps = inps.to(device)

			if activation!=None:
				output = model(inps,activation)
			else:
				output = model(inps)

			for i in range(0,len(inps)):
				print(str(paths[i])+" "+str(torch.argmax(output[i]).item()),file=file)	

def main(path,portion):
	batch_size = 32
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = MyDataset(path,portion)

	train_dataset,validation_dataset = torch.utils.data.random_split(dataset, [len(dataset)*8//10,
		len(dataset)-len(dataset)*8//10], generator=torch.Generator().manual_seed(42))

	train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle=True)

	model = MyModel()
	model.to(device)
	
	epochs = 50
	#0.0003 - 0.000003
	hyperparameters = [0.0003, 0.00019, 0.00012, 0.000076, 0.000048,0.00003]
	layerparameters = [["sigmoid",1024],["sigmoid",1536],["sigmoid",2048],
	["tanh",1024],["tanh",1536],["tanh",2048],
	["relu",1024],["relu",1536],["relu",2048]]
	

	number=1

	for i in range(len(hyperparameters)):
		model.apply(init_normal)
		optimizer = torch.optim.Adam(model.parameters(),lr=hyperparameters[i])
		#torch.save(validation_dataloader, 'dataloader_'+str(i+1)+'.pth')
		train(model,train_dataloader,validation_dataloader,epochs,optimizer,device,batch_size,hyperparameters[i],None,None)
		number+=1

	for i in range(len(hyperparameters)):
		for j in range(len(layerparameters)):
			number+=1
			model = MyModel3(layerparameters[j][1])
			model.to(device)
			model.apply(init_normal)
			optimizer = torch.optim.Adam(model.parameters(),lr=hyperparameters[i])
	#		torch.save(validation_dataloader, 'dataloader_'+str(number)+'.pth')
	#		print(hyperparameters[i],layerparameters[j][0],layerparameters[j][1])
			modelname = "3layered_"+str(hyperparameters[i])+"_"+str(layerparameters[j][0])+str(layerparameters[j][1])
			train(model,train_dataloader,validation_dataloader,epochs,optimizer,device,batch_size,number,layerparameters[j][0],modelname)


	for i in range(len(hyperparameters)):
		for j in range(len(layerparameters)):
			model = MyModel2(layerparameters[j][1])
			model.to(device)
			model.apply(init_normal)
			optimizer = torch.optim.Adam(model.parameters(),lr=hyperparameters[i])
	#		torch.save(validation_dataloader, 'dataloader_'+str(i*len(layerparameters)+j)+'.pth')
	#		print(hyperparameters[i],layerparameters[j][0],layerparameters[j][1])
			modelname = "2layered_"+str(hyperparameters[i])+"_"+str(layerparameters[j][0])+str(layerparameters[j][1])
			train(model,train_dataloader,validation_dataloader,epochs,optimizer,device,batch_size,i*len(layerparameters)+j,layerparameters[j][0],modelname)
			number+=1


def validator():
	batch_size = 32
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = load_model("saved_model_0.1")
	path = "data"
	portion = "train"
	model.to(device)

	#dataset = MyDataset(path,portion)

	#dataloader = DataLoader(dataset, batch_size = batch_size)
	correct = 0

	dataloader = torch.load("dataloader_1.pth")
	#for inps, targets in dataloader:		
	#	inps = inps.to(device)
	#	targets = targets.to(device)
#
#	#	output = model(inps)
#	#	correct += getCorrect(output,targets)
#
	#print(correct/(len(dataloader)*batch_size))
	print(get_valacc(model,dataloader,batch_size,device))

def generateSummary():
	fileNames = os.listdir("4_layer")
	fileNames.sort()
	fileList = []

	for i in range(len(fileNames)):
		if fileNames[i].endswith(".txt"):
			fileList.append([])

	valueList = []

	valacc = []
	trainacc = []

	valloss = []
	trainloss = []

	for i in range(len(fileNames)):
		for j in fileNames:
			if j.endswith(".txt") and j=="val"+str(i+56)+".txt":
				fileList[i]=j
				break
	for i in fileNames:
		file = open("4_layer/"+i,"r")
		valacc.append([])
		trainacc.append([])
		valloss.append([])
		trainloss.append([])
		for j in file:
			splitted = j.split(" ")
			for k in splitted:
				if k.startswith("valacc"):
					value = k.split(":")[-1]
					valacc[-1].append(float(value))	
				if k.startswith("trainacc"):
					value = k.split(":")[-1]				
					trainacc[-1].append(float(value[:-1]))
				if k.startswith("trainaccloss"):
					value = k.split(":")[-1]				
					trainloss[-1].append(float(value[:-1]))
				if k.startswith("valloss"):
					value = k.split(":")[-1]
					valloss[-1].append(float(value))
	for t in range(len(trainacc)):

		print(max(valacc[t]))
	return trainloss, valloss

###### Usage ######  
#main(path, portion)
#predict(modelpath,path,portion,layersize,hiddensize,activation)
###################

###### Example ######
#main("data", "train")
#predict("3layered_0.0003_sigmoid2048","data","train",3,2048,"sigmoid")
#####################