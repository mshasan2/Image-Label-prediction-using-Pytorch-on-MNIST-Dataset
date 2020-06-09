Image prediction using Pytorch on MNIST Dataset using 3 Layers based on Jovian.ml's Deep Learning with PyTorch: Zero to GANs Course and fast.ai's Course.  
#File created using Kaggle NoteBook and PYTHON 3

# Commented out IPython magic to ensure Python compatibility.
import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# %matplotlib inline 
#To plot the graphs within the Jupyter Notebook.
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F #Contains functions like cross_entropy

#Downloading the MNIST training dataset.The dataset elements are images, so we are transforming them to tensors.
dataset=MNIST(root='data/',train=True,transform=transforms.ToTensor(),download=True)
#len(dataset)

#Downloading the MNIST test dataset.The dataset elements are images, so we are transforming them to tensors.
test_ds=MNIST(root='data',train=False,transform=transforms.ToTensor())
#len(test_ds)

#Splitting the training dataset into training(50,000 images) and validation(10,000 images) datasets.
train_ds,val_ds=random_split(dataset,[50000,10000])

#Creating dataloaders to load data in batches.
batch_size=128
train_loader=DataLoader(train_ds,batch_size,shuffle=True)
val_loader=DataLoader(val_ds,batch_size)
test_loader=DataLoader(test_ds,batch_size)

input_size = (28*28)  # Image is of size 28*28 pixels.
num_classes = 10    # Number of digits to be recognised.
hidden_size=16

#Defining the accuracy function
def accuracy(outputs,labels):
  _,preds=torch.max(outputs,dim=1) #Choosing the index of the element with the highest probability in each output row using torch.max(). _ is used to ignore specific values, here we are ignoring the maximum value as we are concerned only with the index of the maximum element.
  return torch.tensor(torch.sum(preds==labels).item()/len(preds)) # == performs element wise comparison of two tensors (Here, tensors containing Predicted values and actulal values) and returns tensor with 0's at unequal element indexes and 1 at equal element indexes.
                                                                  # Here, torch.sum() returns the number of correctly classifird labels.
                                                                  # Divide it by the number if elements to get the accuracy.

#Creating a custom class by inheriting the nn.Module
class MNISTModel(nn.Module):
  def __init__(self, in_size, hidden_size, out_size):
    super().__init__()
    self.linear1=nn.Linear(input_size,hidden_size) # Hidden layer. Performs Linear Transformation.
    self.linear2=nn.Linear(hidden_size,hidden_size*2) # Hidden layer
    self.linear3=nn.Linear(hidden_size*2,out_size)  # Output Layer

  def forward(self,xb): #nn.Module objects are used as if they are functions (i.e they are callable), but behind the scenes Pytorch will call our forward method automatically.
    xb=xb.view(xb.size(0),-1) #Reshaping the tensor.-1 indicates that you know the no. of columns but don't know the no. of rows.
    #Getting intermediate outputs from hidden layers
    out=self.linear1(xb)
    #Applying Activation function ReLU-Rectified Linear Unit
    out=F.relu(out)
    out=self.linear2(out)
    out=F.relu(out)
    #Getting predictions from output layer
    out=self.linear3(out)
    return out

  def training_step(self,batch): #Finding out the loss in training step
    images,labels=batch
    out=self(images) #Output produced by our model
    loss=F.cross_entropy(out,labels) #Finding the loss by comparing output produced by our model with the actual output/labels 
    return loss

  def validation_step(self,batch): #Finding out the loss and accuracy in validation step
    images,labels=batch
    out=self(images)
    loss=F.cross_entropy(out,labels) #Finding the loss
    acc=accuracy(out,labels)
    return {'Val_loss':loss,'Val_acc':acc} #Returning the validation loss and accuracy in the form of a dictionary.

  def validation_epoch_end(self,outputs):
    batch_losses=[x['Val_loss'] for x in outputs] #outputs is a dictionary here,we are selecting the val_loss key here, while using List Comprehension method to store the result in a list.
    epoch_loss=torch.stack(batch_losses).mean() #Torch.stack() concatenates sequence of tensors along a new dimension, and .mean() calculates the mean - This calculates the loss
    batch_accs=[x['Val_acc'] for x in outputs] #outputs is a dictionary here,we are selecting the val_acc key here, while using List Comprehension method to store the result in a list.
    epoch_acc=torch.stack(batch_accs).mean() #Mean accuracies
    return {'Val_loss':epoch_loss.item(),'Val_acc':epoch_acc.item()} #.item() return a python number from a single valued tensor.That is it returns the value present in tensor as a python number.
  
  def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['Val_loss'], result['Val_acc'])) #printing the loss and accuracy for each epoch

model = MNISTModel(input_size,hidden_size=16,out_size=num_classes)

#Loss before training
for images, labels in train_loader:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break

#Checking if GPU is available, True if available
#torch.cuda.is_available()

#Function to get the default device, if GPU is available return cuda (i.e., Our code uses GPU), else returns cpu (Our code defaults to CPU) 
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
#print(device)

#Function that will move data and model to the chosen device(GPU if available, else CPU)
def to_device(data,device):
    if isinstance(data,(list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data] #Pushing each element of the list or tuple into the GPU
    return data.to(device, non_blocking=True)   #pushing data into the GPU with data = data.to('cuda')
                                                #Non-Blocking allows you to overlap compute and memory transfer to the GPU.

#Creating a wrapper class (class whose object wraps or contains a primitive data types.)DeviceDataLoader.
#To wrap our existing data loaders and move data to selected devices(GPU or CPU).
class DeviceDataLoader():
    def __init__(self,dl,device):#Constructor function
        self.dl=dl #dl-DataLoader
        self.device=device
    def __iter__(self): #Iterator Function
        for b in self.dl: #b-batch
            yield to_device(b,self.device) #yielding a batch of data after moving to device
    def __len__(self): #Used when len() function is called on an object of DeviceDataLoader class and returns No. of batches
        return len(self.dl)

train_loader= DeviceDataLoader(train_loader,device)
val_loader=DeviceDataLoader(val_loader,device)

#Function to evaluate the model
def evaluate(model,val_loader):
  outputs=[model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)

def fit(epochs,lr,model,train_loader,val_loader,opt_func=torch.optim.SGD):#Optimization function used by default is Stochastic Gradient Descent
  history=[]
  optimizer=opt_func(model.parameters(),lr)
  for epoch in range(epochs):
    for batch in train_loader: #Training phase
      loss=model.training_step(batch) #Calls training_step method.
      loss.backward() #Computes the gradient.
      optimizer.step() #Updates the gradient.
      optimizer.zero_grad() #Clearing the gradient so that accumulation of gradients doesn't take place.
    #Validation Phase
    result=evaluate(model,val_loader)
    model.epoch_end(epoch,result)
    history.append(result)
  return history #Returning history for each epoch

model = MNISTModel(input_size,hidden_size=16,out_size=num_classes)

to_device(model,device) #moving the model to the correct device

#Evaluating the model before training
history=[evaluate(model,val_loader)]
print(history)

#Training the model for 10 epochs at learning rate of 0.6
history+=fit(10,0.6,model,train_loader,val_loader)

#Training the model for 10 epochs at learning rate of 0.2
history+=fit(10,0.2,model,train_loader,val_loader)

#Training the model for 10 epochs at learning rate of 0.1
history+=fit(10,0.1,model,train_loader,val_loader)

#Training the model for 10 epochs at learning rate of 0.05
history+=fit(10,0.05,model,train_loader,val_loader)

#Plotting a graph of Loss Versus Number of epochs
losses=[x['Val_loss'] for x in history]
plt.plot(losses,'-x') #-x is marker
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Vs. Number of Epochs')

#Plotting a graph of Accuracy Versus Number of epochs
accuracies=[x['Val_acc'] for x in history]
plt.plot(accuracies,'-x') #-x is marker
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Vs. Number of Epochs')
