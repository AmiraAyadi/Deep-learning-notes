# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

## Setting up your machine learning application

### Train /Dev / Test sets

- When you're starting on a new application, it's almost impossible to correctly guess the right values for all of the hyperparameter  on your first attempt. So, in practice applied machine learning is a highly iterative process of IDEA => CODE => EXPERIMENT (you have an idea, you code it then you get back a result and based on the outcome, you might then refine your ideas and change your choices and maybe keep iterating in order to try to find a better and a better neural network.

- intuitions from one domain or from one application area (speech, structural data, vision, etc.) often do not transfer to other application areas and the best choices may depend on the amount of data you have and other parameters/ hyperparameters, if you use GPU, CPU..

What can make us more efficient is setting up a train / dev/ test sets.

We will try to build a model upon training set then try to optimize hyperparameters on dev set as much as possible.
Then after the model is ready we will try on and evaluate the testing set.

the goal is to split the data into three parts :
- Training set. (the largest set)
- Hold-out cross validation set / Development or "dev" set. - Testing set.

In big data era (if your data >= 1.000.00), we don't use the "best practice" of machine learning by split in terms of 60% train, 20% dev and 20% test. The new ratio can be like 98% train, 1% dev and 1% test.

- Also, make sure that the test set and the dev set come from the same distribution. (using internet images fro training and cam images in dev is not a good idea).

- Not having a test set might be okay. (only a dev set)

### Bias / variance

Bias and Variance is one of those concepts that's easily learned but difficult to master.

let's one explanation of both of them:
- if the model is underfitting, it has a high bias.
- If the model is overfitting, then it has a "hight variance"
- the model is just right if it balance the bias/ variance.

For the problem of cat classification, if we know that a human error for this problem is like nearly 0%,we can say that :

- A high variance (overfitting) example would be:
	- Training error: 1%
	- Dev error: 11%
- A high Bias (underfitting) example would be:
	- Training error: 15%
	- Dev error: 14%
- A high Bias (underfitting) && High variance (overfitting) :
	- Training error: 15%
	- Test error: 30%
- The best( because human error is 0%):
	- Training error: 0.5%
	- Test error: 1%

(Note pour moi : on dit un grand biais parce qu'il y a une grande difference entre les vrai parameters et les parmaeters trouvé mais genre au global. On dit grande variance parce que la courbe en soit varie beaucoup trop pour vraiment coller au point des training examples.)

### Basic Recipe for Machine learning

When building a model ask yourself :

does my algorithm has a high bias ? if so:

- Try a bigger NN (more layers, more units)
- try to train it longer
- try to find a new architecture

does my algorithm has a hight variance ? if so:

- get more data
- try regularization
- find a more appropriate NN architecture. 

 Keep doing that until you fit your data better.

Note that in the older days before deep learning, there was a "Bias/variance tradeoff". We now have tools with deep learning to drive down bias and just drive down bias, or drive down variance and just drive down variance, without really hurting the other thing that much.
So for solving the bias and variance problem its really helpful to use deep learning.

## Regularizing your Neural network

Adding regularization to the NN can help reduce the variance.

here is the L1 matrix norm:

    ||W|| = Sum(|w[i,j]|) # sum of absolute values of all w

here is the L2 matrix norm because of arcane technical math reasons is called Frobenius norm:

    ||W||^2 = Sum(|w[i,j]|^2) # sum of all w squared

Also can be calculated as :

    ||W||^2 = W.T * W if W is a vector

Regularization for logistic regression:
The normal cost function that we want to minimize is: 

    J(w,b) = (1/m) * Sum(L(y(i),y'(i)))

The L2 regularization version:

     J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|^2)

The L1 regularization version: 

    J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum(|w[i]|)

- The L1 regularization version makes a lot of w values become zeros, which makes the model size smaller.
- L2 regularization is being used much more often.
- lambda here is the regularization parameter (hyperparameter)
- Regularization for NN:
The normal cost function that we want to minimize is:

    J(W1,b1...,WL,bL) = (1/m) * Sum(L(y(i),y'(i)))

The L2 regularization version:

    J(w,b) = (1/m) * Sum(L(y(i),y'(i))) + (lambda/2m) * Sum((||W[l]||^2)


To do back propagation (old way):

    dw[l] = (from back propagation)

The new way:

    dw[l] = (from back propagation) + lambda/m * w[l]

So plugging it in weight update step:

    w[l] = w[l] - learning_rate * dw[l]
    = w[l] - learning_rate * ((from back propagation) + lambda/m * w[l])
    = w[l] - (learning_rate*lambda/m) * w[l] - learning_rate * (from back propagation)
    = (1 - (learning_rate*lambda)/m) * w[l] - learning_rate * (from back propagation)

- In practice this penalizes large weights and effectively limits the freedom in your model.
- The new term (1 - (learning_rate*lambda)/m) * w[l] causes the weight to decay in proportion to its size.

## Setting up your optimization problem

Here are some intuitions:

Intuition 1:

- If lambda is too large - a lot of w's will be close to zeros which will make the NN simpler (you can think of it as it would behave closer to logistic regression).
- If lambda is good enough it will just reduce some weights that makes the neural network overfit.

Intuition 2 (with tanh activation function):

- If lambda is too large, w's will be small (close to zero) - will use the linear part of the tanh activation function, so we
will go from non linear activation to roughly linear which would make the NN a roughly linear classifier.
- If lambda good enough it will just make some of tanh activations roughly linear which will prevent overfitting.

Implementation tip: if you implement gradient descent, one of the steps to debug gradient descent is to plot the cost
function J as a function of the number of iterations of gradient descent and you want to see that the cost function J
decreases monotonically after every elevation of gradient descent with regularization. If you plot the old definition of J (no
regularization) then you might not see it decrease monotonically.

### Dropout Regularization

- With dropout, what we're going to do is go through each of the layers of the network and set some probability of eliminating a node in neural network.

- the most common way to implement dropout is a technique called inverted dropout.
- Here how to implement it :

    keep_prob = 0.8 # 0 <= keep_prob <= 1
    l = 3 # this code is only for layer 3
    # the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
    d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob
    a3 = np.multiply(a3,d3) # keep only the values in d3
    # increase a3 to not reduce the expected value of output
    # (ensures that the expected value of a3 remains the same) - to solve the scaling problem
    a3 = a3 / keep_prob


- At test time we don't use dropout. If you implement dropout at test time - it would add noise to predictions.

### Understanding Dropout

Let's gain some better intuition of the dropout regularization.

In the previous video, Andrew gave this intuition that drop-out randomly knocks out units in your network. So it's as if on every iteration you're working with a smaller neural network, and so using a smaller neural network seems like it should have a regularizing effect.

A second intuition from the perspective of a single unit:
a neuron can't rely on only one feature, so it have to spread out weights and that have the effect of shrinking the weights - just like L2.

- it's possible to show that dropout has a similar effect to L2.

Note that dropout can have different keep_prob per layer (when we have a big W matrix and other small for example) the downside of this is we will have hyperparamerters to search. One other alternative might be to have some layers where you apply dropout and some layers where you don't apply dropout and then just have one hyperparameter, which is a keep_prob for the layers for which you do apply dropouts. Also note that the input layer dropout has to be near 1 otherwise, we just eliminate a lot of our features input.

Tips: in CV, researchers use it by default. it's because usually just don't have enough data so you're almost always overfitting and dropout IS a technique to prevent overfitting.

A big downside of dropout : the function J (the cost) it's not well defined anymore. It's will be hard to debug the gradient descent when plotting. So the solution is to turn off the dropout and plot / debug your code by setting the keep_prob to 1 and then run the code again, check if J decrease. You can turn on the drop out once you finish this debug.

### Other regularization methods

In addition to L2 regularization and drop out regularization there are few other techniques to reducing over fitting in your neural network

- data augmentation : get new data can be impossible or expensive. So by modify the train data you can double your training set. 
Example with image data : 
	- flipping horizontally image.
	- take random crop (zoom) in a image.
	- random rotation or distortions.

Of course, because you're training set is now a bit redundant this isn't as good as if you had collected an additional set of brand new independent examples. But you could do this without needing to pay the expense.

- Early stopping : plot the training set and the dev set cost together for each iteration. At some iteration the dev
set cost will stop decreasing and will start increasing. This is when we start overfitting, so we have to pick the point at which the training set error and dev set error are best (lowest training cost with lowest dev
cost). We will conceder these as our best parameters.

The advantage of this technique is that we don't have to search for one more hypermarapeters but Andrew prefers to use L2 regularization instead of early stopping because this technique simultaneously tries to minimize the cost function and not to overfit so instead of using different tools to solve the two problems, you're using one that kind of mixes the two and not doing a very code job at it.

- model ensembles (this was only presented in the previous version of the course.)

## Set up your optimization problem


### Normalizing Inputs

When training a neural network, one of the techniques that will speed up your training is if you normalize your inputs.
To do normalization : 
- get the mean of the training set : 

    mean = (1/m) * sum(x(i))

- Subtract the mean from each input : 

    X = X - mean

- Get the variance of the training set :

    variance = (1/m) * sum(x(i)^2)

Normalize the variance of the data:

     X /= variance

Note that subtract the mean will make your inputs centered around 0 so that their mean is 0. Divide by the variance will make the variance of X1 and X2 are both equal to one (when X = (X1, X2).

![normalization](https://i.ibb.co/DfXWTRQ/image.png)

These steps should be applied to training, dev, and testing sets (but using mean and variance of the train set).

if we normalize the input features, it's because we want to make our cost function J easier and faster to optimize. Indeed :

- If we don't normalize the inputs our cost function will be deep and its shape will be inconsistent (elongated) then
optimizing it will take a long time. (left example)

- But if we normalize it the opposite will occur. The shape of the cost function will be consistent (look more
symmetric like circle in 2D example) and we can use a larger learning rate alpha - the optimization will be faster.

![normalizing](https://i.ibb.co/5rs4g7d/image.png)

So, if your input features came from very different scales, it's important to normalize. If it's not the case, it's does any harm.


### Vanishing / Exploding gradients

- When you're training a very deep network your derivatives or your slopes can sometimes get either very,
very big or very small and this makes training difficult. We call that the Vanishing / Exploding gradients.

To understand the problem, suppose that we have a deep neural network with number of layers L, and all the activation
functions are linear [g(z) = z] and each b = 0.

Then:

    Y_hat = W[L]W[L-1].....W[2]W[1]X

Then, if we have 2 hidden units per layer and x1 = x2 = 1, we result in:

    if W[l] = [1.5 0]
    	      [0 1.5] (l != L because of different dimensions in the output layer)
    Y_hat = W[L] [1.5 0]^(L-1) X = 1.5^L # which will be very large
                 [0 1.5]
    if W[l] = [0.5 0]
              [0 0.5]
    Y' = W[L] [0.5 0]^(L-1) X = 0.5^L # which will be very small
              [0 0.5]

The last example explains that the activations (and similarly derivatives) will be decreased/increased exponentially as a
function of number of layers.

- So If W > I (Identity matrix) the activation and gradients will explode.
- And If W < I (Identity matrix) the activation and gradients will vanish.

Recently Microsoft trained 152 layers (ResNet)! which is a really big number. With such a deep neural network, if your
activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or
really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient
descent will take tiny little steps. It will take a long time for gradient descent to learn anything.
There is a partial solution that doesn't completely solve this problem but it helps a lot - careful choice of how you
initialize the weights.

### Weight Initialization for Deep Networks

A partial solution to the Vanishing / Exploding gradients in NN is better or more careful choice of the random
initialization of weights

In a single neuron (Perceptron model): `Z = w1x1 + w2x2 + ... + wnxn`
So if n_x is large we want W 's to be smaller to not explode the cost.

So it turns out that we need the variance which equals 1/n_x to be the range of W 's. lets say when we initialize W 's like this (better to use with tanh activation):

    np.random.rand(shape) * np.sqrt(1/n[l-1])

or variation of this (Bengio et al.):

    np.random.rand(shape) * np.sqrt(2/(n[l-1] + n[l]))

Setting initialization part inside sqrt to `2/n[l-1]` for ReLU is better:

    np.random.rand(shape) * np.sqrt(2/n[l-1])

Number 1 or 2 in the nominator can also be a hyperparameter to tune (but not the first to start with)

This is one of the best way of partially solution to Vanishing / Exploding gradients (ReLU + Weight Initialization with
variance) which will help gradients not to vanish/explode too quickly.

The initialization in this video is called "He Initialization / Xavier Initialization" and has been published in 2015 paper.

### Numerical approximation of gradients

There is an technique called gradient checking which tells you if your implementation of backpropagation is correct.

- Gradient checking approximates the gradients and is very helpful for finding the errors in your backpropagation
implementation but it's slower than gradient descent (so use only for debugging).

Implementation of this is very simple.
Gradient checking:
- First take `W[1],b[1],...,W[L],b[L]` and reshape into one big vector ( theta )
The cost function will be J(theta)
Then take `dW[1],db[1],...,dW[L],db[L]` into one big vector ( d_theta )
Algorithm:

    eps = 10^-7 # small number
    for i in len(theta):
    d_theta_approx[i] = (J(theta1,...,theta[i] + eps) - J(theta1,...,theta[i] - eps)) / 2*eps

Finally we evaluate this formula:

 `(||d_theta_approx - d_theta||) / (||d_theta_approx||+||d_theta||)` ( || -Euclidean vector norm) and check (with eps = 10^-7):
- if it is < 10^-7 - great, very likely the backpropagation implementation is correct
- if around 10^-5 - can be OK, but need to inspect if there are no particularly big values in d_theta_approx -
d_theta vector
- if it is >= 10^-3 - bad, probably there is a bug in backpropagation implementation

### Gradient checking implementation notes

- Don't use the gradient checking algorithm at training time because it's very slow. Use gradient checking only for debugging.
- If algorithm fails grad check, look at components to try to identify the bug.
- Don't forget to add lamda/(2m) * sum(W[l]) to J if you are using L1 or L2 regularization.
- Gradient checking doesn't work with dropout because J is not consistent.
- You can first turn off dropout (set keep_prob = 1.0 ), run gradient checking and then turn on dropout again.
- Run gradient checking at random initialization and train the network for a while maybe there's a bug which can be seen
when w's and b's become larger (further from 0) and can't be seen on the first iteration (when w's and b's are very
small).

###  Notebook Summary  :

#### Initialization

Initialization is very important :
- The weights W should be initialized randomly to break symmetry
- It is however okay to initialize the biases b to zeros. 
- Symmetry is still broken so long as W is initialized randomly
- Different initializations lead to different results
- Random initialization is used to break symmetry and make sure different hidden units can learn different things
- Don't intialize to values that are too large
- He initialization works well for networks with ReLU activations.

A well chosen initialization can:

-   Speed up the convergence of gradient descent
-   Increase the odds of gradient descent converging to a lower training (and generalization) error.

#### Regularization

**Observations**:

-   The value of λλ is a hyperparameter that you can tune using a dev set.
-   L2 regularization makes your decision boundary smoother. If λλ is too large, it is also possible to "oversmooth", resulting in a model with high bias.

**What is L2-regularization actually doing?**:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

**What you should remember** -- the implications of L2-regularization on:

-   The cost computation:
    -   A regularization term is added to the cost
-   The backpropagation function:
    -   There are extra terms in the gradients with respect to weight matrices
-   Weights end up smaller ("weight decay"):
    -   Weights are pushed to smaller values.

#### Dropout

**Note**:

-   A **common mistake** when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
-   Deep learning frameworks like [tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/dropout), [PaddlePaddle](http://doc.paddlepaddle.org/release_doc/0.9.0/doc/ui/api/trainer_config_helpers/attrs.html), [keras](https://keras.io/layers/core/#dropout) or [caffe](http://caffe.berkeleyvision.org/tutorial/layers/dropout.html) come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.

**What you should remember about dropout:**

-   Dropout is a regularization technique.
-   You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
-   Apply dropout both during forward and backward propagation.
-   During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.


**What you should remember from gradient check **:


-   Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
-   Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.
