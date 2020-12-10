
# Neural Networks and Deep Learning

## Introduction to Deep Learning

### What is Neural Network ? 

Let's take an example with house price prediction. We can apply a Linear regression on our data set :

![enter image description here](https://i.ibb.co/7yjfk1V/image.png)

Here we know that a price can be negative, so we arrange our function. This looks like a ReLU function.

- RELU stands for rectified linear unit is the most popular activation function right now that makes deep NNs train faster now.

We can think of this function as a simplest neural network. All the neural does is taking the data as input, compute the function then returning the price. 

A bigger NN would be several neuron stock together : 

![enter image description here](https://i.ibb.co/VqmpS4b/01.jpg)

Deep learning is good at finding connection between input data, and it highlight them in the hidden layers. 

- Deep NN consists of more hidden layers (Deeper layers)
- Each Input will be connected to the hidden layer and the NN will decide the connections.

### Supervised learning with Neural Networks

There are different types of NN for supervised learning :

- Standard NN (Useful for Structured data)
- RNN or Recurrent neural networks (Useful in Speech recognition or NLP)
- CNN or convolutional neural networks (Useful in computer vision)
- Hybrid/custom NN or a Collection of NNs types

Also, we can talk about two types of data : structured and unstructured data.
structured : structured database , csv etc.
unstructured : images, audio, video, etc.

### Why is deep learning taking off?

Deep learning is taking because:

 1. We have more and more data
 2. We have more powerful computation capability
 3. We have more algorithms that change the way we deal with NN, that are more fast to compute for example.

Note that for small data we can perform a Linear regression or SVM (Support vector machine) and get performances that will eventually stop growing with the among of data.  For big data a small NN is better that traditional training algorithms and for big data a big NN is better that a medium NN that is better that small NN.


![](https://i.ibb.co/m9412KZ/11.png)

## Neural Networks Basics

### Binary classification

A binary classification problem is for example telling if a image contain a cat or not. Basically, it's having an input X (for an image is just the pixel vector of that image) and predicting if the value y correspond to a 0 or 1 (for our example, 1 is a cat, 0 not a cat).

Here are some notations:

- M is the number of training examples (vectors)
- N_x is the size of the input example (vector)
- N_y is the size of the output vector
- X^1 is the first input vector
- Y^1 is the first output vector
- X = [x^2 x^2.. x^m]
- Y = [y^1 y^2.. y^m]

### Logistic regression

The logistic regression is an algorithm used for classification, whether is a 2 classes classification or a multi-classes one. 

For our example, we talk about the 2 classes classification. What we want is given our input data X, the probability of y=1 that we re gonna to name ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D). 

The parameters of the LR will be W (N_x dimensional vector like X)  and b a real number.

from that how can we get our ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) ? Well we can define :


![](https://latex.codecogs.com/svg.latex?%5CLarge&space;%5Chat%7By%7D%20=%20w%5ET%20%5Ctimes%20x%20%20b)


But that don't gonna work because  linear regression can be > 1or even negative  but a probability can't and we want is the probability of y=1.

So what we are going to do instead, is using a sigmoid function on ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D).

![](https://latex.codecogs.com/svg.latex?%5CLarge&space;%5Chat%7By%7D%20=%20%5Csigma%28w%5ET%20%5Ctimes%20x%20%20b%29)

With he formula of the sigma function :

![](https://latex.codecogs.com/svg.latex?%5CLarge&space;%5Csigma%28z%29%20=%20%5Cfrac%7B1%7D%20%7B1%20%20%20+e%5E%7B-z%7D%7D)

So when we implement a LR, our job is to find the parameters W and b that make ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) a good estimation of the probability that y be equal to 1. 

### Logistic regression cost function

Now we can see what Loss function we are going to use to measure the performance of our algorithm. First loss function would be the **square root error** that we want to minimize: 

![L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2](https://latex.codecogs.com/svg.latex?%5CLarge&space;L%28%5Chat%7By%7D,%20y%29%20=%20%5Cfrac%7B1%7D%7B2%7D%20%28%5Chat%7By%7D%20-%20y%29%5E2)

But we won't use this notation because it leads us to optimization problem which is not convex, means it contains local optimum points when we try to learn the parameters W and b.

This is the function that we will minimize instead:

![L(\hat{y}, y) = - (y \times \log{\hat{y}}+ (1-y) \times \log{1-\hat{y}})](https://latex.codecogs.com/svg.latex?%5CLarge&space;L%28%5Chat%7By%7D,%20y%29%20=%20-%20%28y%20%5Ctimes%20%5Clog%7B%5Chat%7By%7D%7D%20%20%281-y%29%20%5Ctimes%20%5Clog%7B1-%5Chat%7By%7D%7D%29)

The intuition behind it is:

First, keep in mind the log function :

![log function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Logarithm_plots.png/300px-Logarithm_plots.png)

- If y = 1 then ![](https://latex.codecogs.com/svg.latex?L%28%5Chat%7By%7D,1%29%20=%20-log%28%5Chat%7By%7D%29). 
We want L to be as small as possible,  so we need ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) to be the largest. Knowing that ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) is between [0, 1] then ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) biggest value is 1.

- If y = 0 then  ![](https://latex.codecogs.com/svg.latex?L%28%5Chat%7By%7D,0%29%20=%20-log%281-%5Chat%7By%7D%29). We want ![](https://latex.codecogs.com/svg.latex?1-%5Chat%7By%7D) to be the largest so ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D) to be smaller as possible because it can only has 1 value.

The Cost function will be: 

![](https://latex.codecogs.com/svg.latex?%5CLarge&space;J%28w,b%29%20=%20%5Cfrac%7B1%7D%7Bm%7D%20%5Ctimes%20%5CSigma%7BL%28%5Chat%7By%7D%5Ei,y%5Ei%29%7D)

While the loss function computes the error for a single training example, the cost function is the average of the loss functions of the entire training set.

### Gradient Descent 

The loss function measures how well our algorithms outputs ![](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D)  compares to the ground true label y on each of the training examples. We define before that ![](https://latex.codecogs.com/svg.latex?%5CLarge&space;%5Chat%7By%7D%20=%20%5Csigma%28w%5ET%20%5Ctimes%20x%20%20b%29)
So the cost function measures how well our parameters w and b are doing on the training set.

In order to learn the set of parameters w and b it seems natural that we want want to find w and b that minimize the cost function. Our function is convex (that is in fact one of the main reason we used that function).

First we initialize w and b to 0,0 or initialize them to a random value in the convex function and then try to improve the values the reach minimum value.

In Logistic regression people always use 0,0 instead of random.

The gradient decent algorithm repeats: 

![w = w - alpha * dw](https://latex.codecogs.com/svg.latex?w%20=%20w%20-%20%5Calpha%20%5Ctimes%20dw) 

where :

- alpha is the learning rate, it control how bigger step we take at each iteration on our gradient descent.  
- dw is the derivative of w,  that the modification we want to make to w.

Note : Remember that the definition of a derivative is the slope of a function at the point. The slope of the function is really the height divided by the width.

No mater where we initialize our parameters, we end up getting their value that minimize the loss/ cost function because the derivative give us the direction to find it (L) global minimum.

A we have here two parameters, we will implement:

![w = w - alpha * dw](https://latex.codecogs.com/svg.latex?w%20=%20w%20-%20%5Calpha%20%5Ctimes%20dw) (how much the function slopes in the w direction)

![b = b - alpha * d(J(w,b) / db)](https://latex.codecogs.com/svg.latex?b%20=%20b%20%20-%20%5Calpha%20%5Ctimes%20db) (how much the function slopes in the d direction)


### Computation graph

The computations of a neural network are organized in terms of a forward propagation step, in which we compute the output of the neural network, followed by a backward pass or back propagation step, which we use to compute gradients or compute derivatives. The computation graph explains why it is organized this way.

let's take an example :

![](https://i.ibb.co/ggyxHkp/02.png)
### Derivatives with a Computation Graph

The big thing to remember here is the chain rule.
The chain rule tells us how to find the derivative of a composite function. The rule says (by taking the example of the function J): 
if U effect V effect J when  change is made in U, then,  ![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%28J%29%7D%20%7Bd%28u%29%7D%20=%20%5Cfrac%7Bd%28J%29%7D%20%7Bd%28v%29%7D%20%5Ctimes%20%5Cfrac%7Bd%28v%29%7D%20%7Bd%28u%29%7D)
### Logistic Regression Gradient Descent

Here we are going to see the key equations we need in order to implement gradient descent for logistic regression. We will do the computation using the computation graph.

Let's start by setting our example. 

X1 and X2 are two features (with our first example of house price prediction, these can be the zip code and the year of construction of the house)
For our LR, we need parameters so that :

![z](https://i.ibb.co/yVkp3Xw/image.png)

and :

![enter image description here](https://latex.codecogs.com/svg.latex?%5Chat%7By%7D%20=%20a%20=%20%5Csigma%28z%29)
As we already said, we want to find the right parameters (by changing them)  to minimize our loss function. The computation graph for the forward pass is :

![](https://i.ibb.co/126ByMr/image.png)
After computing our L, we want to minimize it, so we do a backward pass. 
so we calculate "da" which help us get "dz"which help us get in turn "dw1", "dw2" and "db". The final step will be to update w1, w2 and b (gradient descent) to minimize L.

That the computation graph for one example. For all examples in m, we can write a pseudo code :

    J = 0; dw1 = 0; dw2 =0; db = 0; 
    w1 = 0; w2 = 0; b=0; 
    for i = 1 to m:
	    # Forward pass
	    z(i) = W1*x1(i) + W2*x2(i) + b
	    a(i) = Sigmoid(z(i))
	    J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i))) #cumulative
	    
	    # Backward pass
	    dz(i) = a(i) - Y(i)
	    dw1 += dz(i) * x1(i) #cumulative
	    dw2 += dz(i) * x2(i) #cumulative
	    db += dz(i) #cumulative
    J /= m
    dw1/= m
    dw2/= m
    db/= m
    # Gradient descent
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b = b - alpha * db

Note that the derivative, respect to w1 of the overall cost function is also going to be the average of derivatives respect to w_1 of the individual loss terms: 

![function](https://latex.codecogs.com/svg.latex?%5Cfrac%20%7BdJ%7D%7Bdw_1%7D%20=%20%5Cfrac%7B1%7D%7Bm%7D%20%5CSigma%7B%5Cfrac%20%7Bd%20L%28a%5Ei,%20y%5Ei%29%7D%7Bdw%5Ei_1%7D%7D)

We know how to compute w1(i) on a single training example, so, what we need to do is just compute these derivatives and average them. That why dw1 and dw2 are cumulative and that why we average them later on the pseudo code.

Note also that we need one more loop if we have more features as we often do in ML/DL. So vectorization is really important to reduce the loops.


### Vectorization

Vectorization is basically the art of getting rid of explicit loops in your code.
We use more and more data in our algorithms and it's important that our code run quickly to see the results (unless you love to wait). That is where vectorization come in. 

So it's important to avoid explicit for loop whenever that's possible. There are multiples functions on numpy for example that use vectorization by default. Whether you are using GPU or CPU. 

#### Vectorizing Logistic Regression

- First example to rid off the one possible loop in our first pseudo code, is to change our variables dw_i by using instead one vector W.

That way :

     dw1 = 0; dw2 =0;
become :

    dw = np.zeros((n_x, 1))

- By the same logic, we can replace the first for loop by just two lines of code :

     for i = 1 to m:
    	    # Forward pass
    	    z(i) = W1*x1(i) + W2*x2(i) + b
    	    a(i) = Sigmoid(z(i))

become:

    Z = np.dot(W.T,X) + b
    A = 1 / 1 + np.exp(-Z)

How's that ? we use an input X, a  (n_x, m) matrix to compute Z as :
[z1,z2...zm] = W^t * X + [b,b,...b]

A is just compute by applying the sigma function to Z. the whole thing using vectorization by numpy.

Note that in python the b is a real number, python automatically change it to a (1, m) matrix once it has to add it to W^t * X. 

We call that broadcasting : In general if you have an (m,n) matrix and you add(+) or subtract(-) or multiply(*) or divide(/) with a (1,n) matrix, then this will copy it m times into an (m,n) matrix. The same with if you use those operations with a (m , 1) matrix, then this will copy it n times into (m,n) matrix. And then apply the addition, subtraction, and multiplication of division element wise.

Finally, let's "vectorize" the reminding code:

    dz(i) = a(i) - Y(i)

can simply become :

    dz = A - Y
As for the :

    db += dz(i)

become :

    db = dz.sum() / m
    
The final implementation look like this :

    Z = np.dot(W.T,X) + b 
    A = 1 / 1 + np.exp(-Z)
    dz = A - Y # dz shape is (1, m)
    dw = np.dot(X, dz.T) / m # dw shape is (Nx, 1)
    db = dz.sum() / m
    w = w - alpha * dw
    b = b - alpha * db

This is just ONE step of gradient descent. There is no way of get rid of a for loop if we want to compute several ones.
Also  recall that "np.dot(a,b)" performs a matrix multiplication on a and b, whereas "a*b" performs an element-wise multiplication. 

Here some tricks to have a bug free code :
- Always specify the shape of your vector to (m, 1) to avoid the shape (m, ).
- don't be afraid to use assert in your code (cheap in calculations) 
- don't be shy about calling the reshape operation to make sure that your matrices or your vectors are the dimension that you need it to be.

Notebook's notes :

- The main steps for building a Neural Network are:

1.  Define the model structure (such as number of input features)
2.  Initialize the model's parameters
3.  Loop:
    -   Calculate current loss (forward propagation)
    -   Calculate current gradient (backward propagation)
    -   Update parameters (gradient descent)

Build 1-3 separately and integrate them into one function we call `model()`.

- Gradient descent converges faster after normalization of the input matrices.

Note : To perform the classification, here are the step we implement :
- reshape images to have (64*64*3, m) matrix where m = 209.
- initialize the parameters then propagate forward and backward (calculate the loss, calculate the gradient, calculate the upgraded  parameters)
- predict the result to compute the accuracy

- In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate αα determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

- A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.

Finally : 

**What to remember from the 1st assignment:**

2.  Preprocessing the dataset is important.
3.  You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
4.  Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. You will see more examples of this later in this course!



## Shallow Neural Networks

We had talked about logistic regression and we saw how this model corresponds to the following computation draft:

![lr_graph](https://i.ibb.co/bvGbM2b/image.png)


If we have one hidden layer added, we will have :

![lr_layer_graph](https://i.ibb.co/vcg88YW/tempsnip.png)

This will be the new notation of the section : [1] refer to the 1st computation in the first layer and [2] refer to the 2nd computation in the second layer.

Let's talk about representation :

a NN contains  input layers, some hidden layers and a output layers. The hidden layers are called like that because we can't see what are their true values in the training set as we do for the input and output layers. 

Introduction of some additional notations :

- Whereas previously, we were using the vector X to denote the input features an alternative notation will be a^[0].

Note that the A also stands for activations, and it refers to the values that different layers of the neural network are passing on to the subsequent layers. The input layer passes on the value x to the hidden layer, so we're going to call that "activations of the input layer a^[0]".

- a^[1] will represent the activation of the hidden neurons.

In literature and when we're talking, we say that we have (total layer - input layer) Layers NN. So in the image above, we have a 2 layers NN even taught we see 3 layers.

### Computing a Neural Network's Output

We've said before that the circle in logistic regression, really represents two steps of computation rows. First we compute z  and second we compute the activation as a sigmoid function of z.

Here are the 4 equations that we need to compute a 2 layers NN output. For each circle, the compute representation is display:

![](https://i.ibb.co/7Xxkdv3/image.png)
We can note that :

- We have 3 features x, so N_x=3 and we have 4 neurons in the hidden layer. let's call that n_h.
- In the first hidden layer, in each neuron, we compute the different z^[1], the result of "w.T * x + b " (index notation are here ignored) and the different a^[1], the sigmoid function applied to z^[1].
- We can vectorize this by stack together all the z^[1]_[i] into z^[1] and all the a^[1]_[i] into a^[1].
- z^[1] shape is (n_h, 1).
- a^[1] shape is (n_h, 1).
- for compute z^[1], we use W^[1] and b^[1] they are respectively all the weight used in the first hidden layer stack together in a matrix of shape (n_h, n_x) and  all the bias used in the first hidden layer stack together in a matrix of shape (n_h, 1) .
- We do the exact same steps for the output layer and we get z^[2] and a^[2].


### Vectorizing across multiple examples

We already show how to compute the output with one example for the 2 layers NN. For multiples example, we can do in pseudo code:

    for i = 1 to m:
    z[1, i] = W1*x[i] + b1 # shape of z[1, i] is (n_h,1)
    a[1, i] = sigmoid(z[1, i]) # shape of a[1, i] is (n_h,1)
    z[2, i] = W2*a[1, i] + b2 # shape of z[2, i] is (1,1)
    a[2, i] = sigmoid(z[2, i]) # shape of a[2, i] is (1,1)

We use a for loop in the preview pseudo code, as we said, we want to get rid of all the for-loop that we can replace by vectorization. 

So to vectorize the for loop :

    Z1 = W1X + b1 # shape of Z1 (n_h,m)
    A1 = sigmoid(Z1) # shape of A1 (n_h,m)
    Z2 = W2A1 + b2 # shape of Z2 is (1,m)
    A2 = sigmoid(Z2) # shape of A2 is (1,m)
    
Note that X = A[0], so we end up with two similar equations, with just the index that increase.

### Activation functions

One choose we have to made when building NN is to choose the activation function in the hidden layers. We only use sigmoid functions far, but other function can work better.

One other activation function can be the tanh function, which is a shifted version of the sigmoid function (crossing at 0 instead of  0.5). The range of this function is [-1, 1]. One implementation of the function can be (otherwise, we can just use the np.tanh() function) :

    A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) 

Notice that this function can (most of the time) work better that the sigmoid activation function(exception for the output layer in binary classification) . In fact, the mean of its output is closer to zero, and so it centers the data better for the next layer. We will talk with more detail about that fact later. 

One of the downsides of both the sigmoid function and the tanh function is that if z is either very large or very small, then the gradient or the derivative or the slope of this function becomes very small. So if z is very large or z is very small, the slope of the function ends up being close to 0. And so this can slow down gradient descent.

One of the popular activation functions that solved the slow gradient decent is the RELU function. The formula :
`RELU = max(0,z)` 
 If z is negative the slope is 0 and if z is positive the slope remains 1. If = 0, we can pretend it's either 0 or 1 because the the derivative is not well defined. One of it downsides is that the derivative is always equal to 0 when z is negative so there is a another version of it : the Leaky  ReLU.
 
The leaky ReLU is slightly tilted downwards instead of being equal to 0 when z is negative. This usually works better than the ReLU activation function, although it's just not used as much in practice.

The advantage of both the ReLU and the leaky ReLU is that a large part of the derivative is very different from 0. And so in practice, using the ReLU activation function, the neural network will often learn much faster than when using the tanh or the sigmoid activation function.

To recap, we have the choice between : 

- the sigmoid function (don't use unless it's in the output layer and you have a binary classification problem) 
- the tanh function (better that sigmoid function)
- the ReLU function (mostly used)
- the Leaky ReLU function (we can try this).

Activation functions is not the only choice we have to build NN (the numb of layers, of neurons, the value of the learning rate etc.). The best thing to do is to try and evaluate.

### Why do you need non-linear activation functions

Why do we need an activation function ? let' see !

If we removed the activation function from our algorithm that can be called linear activation function or the identity activation function.

- If we use linear activation function (so if we remove the activation function) it will output linear activations and whatever hidden layers you add, the activation will be always linear. So it's just like you didn't add any hidden layer.

- You might use linear activation function in one place - in the output layer if the output is real numbers (regression problem). But even in this case if the output value is non-negative you could use ReLU instead.

### Derivatives of activation functions

Here are the derivatives of all of our activation functions :

Derivation of Sigmoid activation function:

    g(z) = 1 / (1 + np.exp(-z))
    g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
    g'(z) = g(z) * (1 - g(z))

Derivation of Tanh activation function:

    g(z) = (e^z - e^-z) / (e^z + e^-z)
    g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2

Derivation of RELU activation function:

    g(z) = np.maximum(0,z)
    g'(z) = { 0 if z < 0
    1 if z >= 0 }

Derivation of leaky RELU activation function:

    g(z) = np.maximum(0.01 * z, z)
    g'(z) = { 0.01 if z < 0
    1 if z >= 0 }

### Gradient descent for Neural Networks

In this video, we will see how to implement gradient descent for the neural network with one hidden layer. The equations are given.

We have : 
- n[0] = Nx
- n[1] = the number of hidden neurons
- n[2] = the number of output neurons = 1
- W1 shape is (n[1],n[0])
- b1 shape is (n[1],1)
- W2 shape is (n[2],n[1])
- b2 shape is (n[2],1)

- Cost function :

    I = I(W1, b1, W2, b2) = (1/m) * Sum(L(Y,A2))

- Then Gradient descent:

    Repeat:
    Compute predictions (y'[i], i = 0,...m)
    Get derivatives: dW1, db1, dW2, db2
    Update: W1 = W1 - LearningRate * dW1
    b1 = b1 - LearningRate * db1
    W2 = W2 - LearningRate * dW2
    b2 = b2 - LearningRate * db2

- Forward propagation:

    Z1 = W1A0 + b1 # A0 is X
    A1 = g1(Z1)
    Z2 = W2A1 + b2
    A2 = Sigmoid(Z2) # Sigmoid because the output is between 0 and 1

- Backpropagation (derivations):

    dZ2 = A2 - Y # derivative of cost function we used * derivative of the sigmoid function
    dW2 = (dZ2 * A1.T) / m
    db2 = Sum(dZ2) / m
    dZ1 = (W2.T * dZ2) * g'1(Z1) # element wise product (*)
    dW1 = (dZ1 * A0.T) / m # A0 = X
    db1 = Sum(dZ1) / m # Hint there are transposes with multiplication because to keep dimensions correct

### Random Initialization 

We said earlier that we can initialize the weights in the logistic regression at 0. In NN, we have to initialize them randomly. Indeed,if we do that (initialize at 0) it will not work. Let's see why !

- a W initialize at 0 means that all hidden units will be completely identical and compute exactly the same function. So having multiple hidden units or one lead to the same result.

- The solution to this is to initialize the parameters randomly. We can set 

    w1 = np.random.randn((2,2,))

This generates a gaussian random variable (2,2). And then we multiply this by very small number, such as 0.01.

For b, it turns out that b does not have what's called the symmetry breaking problem. So it's okay to initialize b to just zeros.

We choose a small W because of the problem we talked earlier for the tanh() and the sigmoid function : if the weight is too large we will end up at the very start of training with very large values of Z. Which causes the tanh or the sigmoid activation function to be saturated, thus slowing down learning.

Note that constant 0.01 is alright for 1 hidden layer networks, but if the NN is deep, it's better to use another small number.


## Deep Neural Networks

### Deep L-layer neural network

- We say that logistic regression is a very "shallow" model, whereas a model with 5 hidden layers for example is a much deeper model.

- So shallow versus depth is a matter of degree.

Notation we're going to use :

- L to denote the number of layers in the network.
- n^[l] #unit (neurons) in layer l. So n^[0] is the number of input features and n^[L] is the number of neurons in the output layer.
-  a^[l] denote the activation in layer l and a^[l] = g^[l](z^[l])
- We use w^[l] to compute z^[l]
- x= a^[0] and a[l] = y.

### Forward Propagation in a Deep Network

We already said that the back propagation for one layer l and for one input example is :

    z^[l] = W^[l]a^[l-1] + b^[l]
    a^[l] = g^[l](a^[l])

for m input example , we will have:

    Z^[l] = W^[l]A^[l-1] + B^[l]
    A^[l] = g^[l](A^[l])

We will have to compute this for each layer l using a for loop because there is no way to do it without it.

## Getting your matrix dimensions right

One important thing to do to have debug a code (or to have a bug free) is to figure out your matrix dimension. Let's how.

first thing is not to be shy by debugging using a pencil and a paper and remember the rules of matrix calculus, using that we can find that :

- w^[l] will always have dimension of (n^[l], n^[l-1]).
- b^[l] will always be a (n^[l], 1) dimension matrix.
- dw^[l] have the same dimension as w^[l].
- db^[l] have the same dimension as b^[l].
- Dimension of Z^[l], A^[l] , dZ^[l] , and dA^[l] is (n^[l],m) in a vectorize example where m is the number of example.

### Why deep representation ?

Here we are going to discuss why are deep neural networks so effective, and why do they do better than shallow representations?

We can think of the earlier layers of the neural network as detecting simple functions and then composing them together in the later layers of a neural network so that it can learn more and more complex functions., So deep NN makes relations with data from simpler to complex. 
We saw to examples : 

1. Face recognition application:
Image ==> Edges ==> Face parts ==> Faces==> desired face

2. Audio recognition application:
Audio ==> Low level sound features==> Phonemes ==> Words ==> Sentences

Neuroscientist believe that the human brain also starts off detecting simple things like edges then builds those up to detect more complex things like faces. ( analogies between DL and human brain are sometimes dangerous).

The other piece of intuition about why deep networks seem to work well is the circuit theory :

there are functions we can compute with a relatively small but deep neural network (the number of hidden units is relatively small) but if we try to compute the same function with a shallow network, so if there aren't enough hidden layers, then we might require exponentially more hidden units to compute.

Tip : when building a model start with something simple like logistic regression then try something with one or two hidden layers and use that as a hyper parameter.
Use that as a parameter or hyper parameter that you tune in order to try to find the right depth for your neural network.

### Building blocks of deep neural networks

![forward and backward prop](https://i.ibb.co/7kdQKBK/08.png)

### Forward and Backward Propagation

How to implement the previous steps  of forward and backward prop ?

for forward prop :

    Input A[l-1]
    Z[l] = W[l]A[l-1] + b[l]
    A[l] = g[l](Z[l])
    Output A[l], cache(Z[l])

for backward :

    Input da[l], Caches
    dZ[l] = dA[l] * g'[l](Z[l]) #element wise product
    dW[l] = (dZ[l]A[l-1].T) / m
    db[l] = sum(dZ[l])/m # Dont forget axis=1, keepdims=True
    dA[l-1] = w[l].T * dZ[l] # The multiplication here are a dot product.
    Output dA[l-1], dW[l], db[l]

for loss function :

    dA[L] = (-(y/a) + ((1-y)/(1-a)))

### Parameters vs Hyperparameters


We know that the main parameters of the NN is W and b. But we also need other parameters that will "control" our NN. we call these parameters "hyper parameters" and they are :
- The learning rate
- the number of iteration 
- the number of hidden layer
- the number of hidden units
- the activation function that we choose

They control the main parameters that why we called them hyper parameters. 
We will see other HP later such as momentum, mini batch size  etc.

To find the best hyper parameters, we have to try different values and choose the best.

Note that even if we tune our algorithm with the best hyper parameters, but one year from now, it can change. 


