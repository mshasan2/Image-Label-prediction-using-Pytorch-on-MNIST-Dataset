#Image prediction using Pytorch on MNIST Dataset based on Jovian.ml's Deep Learning with PyTorch: Zero to GANs Course and fast.ai's Course.  
#File created using Google Colab and PYTHON 3

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

input_size = 28*28  # Image is of size 28*28 pixels.
num_classes = 10    # Number of digits to be recognised.
lr = 0.001

#Defining the accuracy function
def accuracy(outputs,labels):
  _,preds=torch.max(outputs,dim=1) #Choosing the index of the element with the highest probability in each output row using torch.max(). _ is used to ignore specific values, here we are ignoring the maximum value as we are concerned only with the index of the maximum element.
  return torch.tensor(torch.sum(preds==labels).item()/len(preds)) # == performs element wise comparison of two tensors (Here, tensors containing Predicted values and actulal values) and returns tensor with 0's at unequal element indexes and 1 at equal element indexes.
                                                                  # Here, torch.sum() returns the number of correctly classifird labels.
                                                                  # Dividing it by the number of elements to get the accuracy.

#Creating a custom class by inheriting the nn.Module
class MNISTModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear=nn.Linear(input_size,num_classes) #Performs Linear Transformation.

  def forward(self,xb): #nn.Module objects are used as if they are functions (i.e they are callable), but behind the scenes Pytorch will call our forward method automatically.
    xb=xb.reshape(-1,784) #reshape() returns a tensor with the same data and same number of elements, but with the specified shape.
    out=self.linear(xb)
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

model=MNISTModel()

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

print('Loss before training{}'.format(evaluate(model,val_loader)))

#Running the model for epochs
history = fit(100, 0.001, model, train_loader, val_loader)

#To plot a graph of Accuracy vs Number of epochs
accuracies = [r['Val_acc'] for r in history]
plt.plot(accuracies, '-x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of epochs of epochs');

result = evaluate(model, test_loader)
print(result)

def predict_image(img, model):
    xb = img.unsqueeze(0) #Adding another dimension to the tensor
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

#Predicting the image label on Test Dataset
img, label = test_ds[518]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))

