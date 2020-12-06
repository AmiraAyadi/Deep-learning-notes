
# Neural Networks and Deep Learning

## Introduction to Deep Learning

### What is Neural Network ? 

Let's take an example with house price prediction. We can apply a Linear regression on our data set :

![enter image description here](https://i.ibb.co/7yjfk1V/image.png)

Here we know that a price can be negative, so we arrange our function. This Looks like a ReLU function.

- RELU stands for rectified linear unit is the most popular activation function right now that makes deep NNs train faster
now.

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

Deep learning is taking beause:

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

With the formula of the sigma function :

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
