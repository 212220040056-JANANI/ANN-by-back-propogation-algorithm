### EX NO : 06
### DATE  : 02/05/2022
# <p align="center"> BACK PROPAGATION USING ANN (MLP) </p>
## AIM:
   To implement multi layer artificial neural network using back propagation algorithm.
## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## RELATED THEORY CONCEPTS:
Algorithm for ANN Backpropagation:

• Weight initialization: Set all weights and node thresholds to small random numbers. Note that the node threshold is the negative of the weight from the bias unit(whose activation level is fixed at 1).

• Calculation of Activation:

The activation level of an input is determined by the instance presented to the network. The activation level oj of a hidden and output unit is determined. • Weight training:

Start at the output units and work backward to the hidden layer recursively and adjust weights.

The weight change is completed.

The error gradient is given by:

a. For the output units.

b. For the hidden units.

Repeat iterations until convergence in term of the selected error criterion. An iteration includes presenting an instance, calculating activation and modifying weights.

## ALGORITHM:
1..Import packages

2.Defining Sigmoid Function for output

3.Derivative of Sigmoid Function

4.Initialize variables for training iterations and learning rate

5.Defining weight and biases for hidden and output layer

6.Updating Weights

## PROGRAM:
```
/*
Program to implement ANN by back propagation algorithm.
Developed by   : JANANI D
RegisterNumber :  212220040056
*/

import numpy as np
X = np.array(([2, 9], [1,5], [3,6]), dtype=float )
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) #maximum of X array longitudinally
y = y/100

#Sigmoid Function
def sigmoid (x):
  return 1/(1+np.exp(-x))
#Derivatives of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1-x)
  #Variable initialization
epoch = 7000 #Setting training iterations
lr= 0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neuropns at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh= np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
#draws a random range uniformly of dim x*y
for i in range(epoch):
#Forward Propagation
  hinp1=np.dot(X,wh)
hinp=hinp1 + bh
hlayer_act = sigmoid(hinp)
outinp1=np.dot(hlayer_act,wout)
outinp=outinp1+bout
output = sigmoid(outinp)
#Back propagation
EO = y-output 
outgrad = derivatives_sigmoid(output)
d_output = EO* outgrad
EH = d_output.dot(wout.T)
hiddengrad= derivatives_sigmoid(hlayer_act)
d_hiddenlayer = EH * hiddengrad
wout += hlayer_act.T.dot(d_output) *lr 
#dotproduct of next layer
#bout += np.sum(d_output, axis=0,keepdims=True)
wh += X.T.dot(d_hiddenlayer) *lr
#bh += np.sum(d_hiddenlayer,axis=0,keepdims=True) *lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)
```

## OUTPUT:
![image](https://user-images.githubusercontent.com/86832944/169064784-f7378cbc-ef8c-4c63-b370-f5355383a457.png)


## RESULT:
Thus the python program successully implemented multi layer artificial neural network using back propagation algorithm.
