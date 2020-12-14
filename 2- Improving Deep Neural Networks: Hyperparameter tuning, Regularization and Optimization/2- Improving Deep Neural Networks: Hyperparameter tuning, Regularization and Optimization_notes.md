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
    #the generated number that are less than 0.8 will be dropped. 80% stay, 20% dropped
    d3 = np.random.rand(a[l].shape[0], a[l].shape[1]) < keep_prob
    a3 = np.multiply(a3,d3) # keep only the values in d3
    #increase a3 to not reduce the expected value of output
    #(ensures that the expected value of a3 remains the same) - to solve the scaling problem
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

## Optimization algorithms

### Mini-batch gradient descent

We are able to train neural networks on a huge data set and that training is slow. Having fast optimization algorithms, can really speed up your efficiency. Let's talk about mini-batch gradient descent.

- suppose we have 50 million examples for training, so `m= 50 million`, maybe the training of these huge data won't fit into the memory at once, or will take a lot of time to compute at once, we have to find an other way to train. 
- One faster way to do this is by making the gradient descent process only some of the data at once, or **mini-batches**.
- In our example, suppose we split our m nto mini-batches of 1000, so a new notation would be :

    X^{1} = 0.....1000, where {1] is the mini-batche number, for m=50 milion, there will be 5000 mini batches of 1000 each.

We do the same to Y, so one mini-batches would be : `t = X{t}, Y{t}`
So the difference is that in batch gradient descent, we run the algorithm on the whole dataset at once. While on the mini-batches gradient descent, we run it on the mini-batches.

Here a pesudo code of the implementation of the mini-batches grandient descent :

    for t = 1:No_of_batches 
    	AL, caches = forward_prop(X{t}, Y{t})
    	cost = compute_cost(AL, Y{t})
    	grads = backward_prop(AL, caches)
    	update_parameters(grads)

- we will call one loop over one mini-batches an **epoch** (a single pass trought the training set) and of course, the code inside the loop is vectorized.

Note that this method work more faster in large dataset.

### Understanding mini-batch gradient descent

If we've had the cost function J as a function of different iterations it should decrease on every single iteration.
On mini batch gradient descent though, if you plot progress on your cost function, then it may not decrease on every iteration. It could contain some ups
and downs but generally it has to go down.

If the mini-batches = m, so we took the whole dataset, and that just **batch gradient descent**. 

If the mini-batches = 1, then we took every single example of X and Y, and run the gradient descent on them. This is the **stochastic gradient descent**

The best thing to do is to choose a mini-batche size between 1 and m. That is the **mini-batches gradient descent**.

Here is the difference between each of these three gradient descent :

- Batch gradient descent:
	- too long per iteration (epoch)
- Stochastic gradient descent:
	- too noisy regarding cost minimization (can be reduced by using smaller learning rate)
	- won't ever converge (reach the minimum cost)
	- lose speedup from vectorization
- Mini-batch gradient descent:
	- faster learning: you have the vectorization advantage make progress without waiting to process the entire                    training set
	- doesn't always exactly converge (oscelates in a very small region, but you can reduce learning rate)

So one question can be how to choose this mini-batches size?

If we have a small training set (< 2000 examples) -> use batch gradient descent.

It your dataset is bigger, use a mini batches that (it has to be a) is  power of 2 (because of the way computer memory is layed out and accessed, sometimes your code
runs faster if your mini-batch size is a power of 2) so choose for example: 64, 128, 256, 512, 1024, ...

Finally, make sure that mini-batch fits in CPU/GPU memory.

Note that the mini-batches size can be added to the list of the hyperparameters to search and tune.

### Exponentially weighted averages

To talk about the other algorithms that work better than the gradien descent, we need to fisrt learn what is the Exponentially weighted averages and how it works. 

let's illustrate this but ploting the temperature of days trough the year. We can see that they the temp. is small in winter and big in the summer but mostly that the data is noisy.

![noisy](https://i.ibb.co/fNSxrjC/image.png)

To compute the Exponentially weighted averages, we can do:

    V0 = 0
    V1 = 0.9 * V0 + 0.1 * t(1) = 4 # 0.9 and 0.1 are hyperparameters
    V2 = 0.9 * V1 + 0.1 * t(2) = 8.5
    V3 = 0.9 * V2 + 0.1 * t(3) = 12.15
    ...

That lead us to a general equation : 

    V(t) = beta * v(t-1) + (1-beta) * theta(t)

If we plot this it will represent averages over ~ (1 / (1 - beta)) entries:

- beta = 0.9 will average last 10 entries
- beta = 0.98 will average last 50 entries
- beta = 0.5 will average last 2 entries

### Understanding exponentially weighted averages

What the EWA really doing ?

The intuition of that equation is that we take all the theta-(t-1) and replace it by the value of the equation for that value :

![intuition](https://i.ibb.co/V2S4gkB/image.png)

- We can implement this algorithm in pseudo code:

    v = 0
    Repeat
    {
	    Get theta(t)
	    v = beta * v + (1-beta) * theta(t)
    }

- We can compute a moving window, where you explicitly sum over the last 10 days, the last 50 days temperature and just divide by 10 or divide by 50 and that usually gives you a better estimate. But the disadvantage of that is explicitly keeping all the temperatures around and sum of the last 10 days is it requires more memory and it's just more complicated to implement and is computationally more expensive. This is a very efficient way to do so both from computation and memory efficiency point of view which is why it's used in a lot of machine learning. Not to mention that there's just one line of code which is, maybe, another advantage.

### Bias correction in exponentially weighted averages

- The first value of the algorithm is very pooorly estimated because of the initial phase `v0= 0`.
- The bias correction helps make the exponentially weighted averages more accurate.
- To solve the bias issue we have to use this equation:

    v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)

As t becomes larger the `(1 - beta^t)` becomes close to 1.


### Gradient descent with momentum

There's an algorithm called momentum, or gradient descent with momentum that almost always works faster than the standard gradient descent algorithm.

the basic idea is to compute an exponentially weighted average of your gradients, and then use that gradient to update your weights instead. Let's unpack that and see how we can actually implement this.

Pseudo code :

    vdW = 0, vdb = 0
    on iteration t:
	    # can be mini-batch or batch gradient descent compute dw, db on current mini-batch
	    vdW = beta * vdW + (1 - beta) * dW
	    vdb = beta * vdb + (1 - beta) * db
	    W = W - learning_rate * vdW
	    b = b - learning_rate * vdb

- Momentum helps the cost function to go to the minimum point in a more fast and consistent way.
- beta is another hyperparameter . Note that beta = 0.9 is very common and works very well in most cases.

- In practice people don't bother implementing bias correction when implementing gradient descent with momentum.

- Some paper will omit the (1 - beta) but Andrew prefer the first version. It's more intuitive.

### RMSprop

There's another algorithm called RMSprop, which stands for **root mean square prop**, that can also speed up gradient descent. Let's see how it works.

The pseudo code : 

    sdW = 0, sdb = 0
    on iteration t:
	    # can be mini-batch or batch gradient descent compute dw, db on current mini-batch
	    sdW = (beta * sdW) + (1 - beta) * dW^2 	# squaring is element-wise
	    sdb = (beta * sdb) + (1 - beta) * db^2 	# squaring is element-wise
	    W = W - learning_rate * dW / sqrt(sdW)
	    b = B - learning_rate * db / sqrt(sdb)

Let's say that b is the vertical direction and w the horizental direction. RMSprop will make the cost function move slower on the b direction and faster on the w direction.

- Ensure that sdW is not zero by adding a small value epsilon (e.g. epsilon = 10^-8 ) to it:

    W = W - learning_rate * dW / (sqrt(sdW) + epsilon)

With RMSprop you can increase your learning rate.

One fun fact about RMSprop, it was actually first proposed not in an academic research paper, but in a Coursera course that Jeff Hinton had taught on Coursera many years ago.

### Adam optimization algorithm

- The Adam optimization algorithm is basically taking momentum and rms prop and putting them together. Adam optimization and RMSprop are among the optimization algorithms that worked very well with a lot of NN architectures.

Here the pseudo code :

    vdW = 0, vdW = 0
    sdW = 0, sdb = 0
    on iteration t:
	    # can be mini-batch or batch gradient descent
	    compute dw, db on current mini-batch
	    vdW = (beta1 * vdW) + (1 - beta1) * dW # momentum
	    vdb = (beta1 * vdb) + (1 - beta1) * db # momentum
	    sdW = (beta2 * sdW) + (1 - beta2) * dW^2 # RMSprop
	    sdb = (beta2 * sdb) + (1 - beta2) * db^2 # RMSprop
	    vdW = vdW / (1 - beta1^t) # fixing bias
	    vdb = vdb / (1 - beta1^t) # fixing bias
	    sdW = sdW / (1 - beta2^t) # fixing bias
	    sdb = sdb / (1 - beta2^t) # fixing bias
	    W = W - learning_rate * vdW / (sqrt(sdW) + epsilon)
	    b = B - learning_rate * vdb / (sqrt(sdb) + epsilon)


Hyperparameters for Adam:
- Learning rate: needed to be tuned.
- beta1 : parameter of the momentum - 0.9 is recommended by default.
- beta2 : parameter of the RMSprop - 0.999 is recommended by default.
- epsilon : 10^-8 is recommended by default.

- Adam Stands for Adaptive Moment Estimation.

### Learning rate decay

One of the things that might help speed up your learning algorithm, is to slowly reduce your learning rate over time. We call this learning rate decay.

As mentioned before mini-batch gradient descent won't reach the optimum point (converge). But by making the learning rate decay with iterations it will be much closer to it because the steps (and possible oscillations) near the optimum are smaller.

One technique equations is :

`learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate_0`

Note that the decay rate here becomes another hyper-parameter, which you might need to tune.

Other learning rate decay methods :
- `learning_rate = (0.95 ^ epoch_num) * learning_rate_0` which is the exponentially decay
- `learning_rate = (k / sqrt(epoch_num)) * learning_rate_0`

Some people are making changes to the learning rate manually.

For Andrew Ng, learning rate decay has less priority.

### The problem of local optima

In the early days of deep learning, people used to worry a lot about the optimization algorithm getting stuck in bad local optima. But as this theory of deep learning has advanced, our understanding of local optima is also changing.

- The normal local optima is not likely to appear in a deep neural network because data is usually high dimensional. For point to be a local optima it has to be a local optima for each of the dimensions which is highly unlikely.
- It's unlikely to get stuck in a bad local optima in high dimensions, it is much more likely to get to the saddle point rather to the local optima, which is not a problem.
- The problem can be the plateaus, they can make the learning very very slow. They are a region where the derivative is close to 0 for a long time (like an horizental zone). In such situation, the algorithm that we learn earlier can do a good job. They speed up the rate rate at which you could move down the plateau and then get off the plateau.

**What you should remember**:

-   The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is the number of examples you use to perform one update step.
-   You have to tune a learning rate hyperparameter αα.
-   With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).
-  Shuffling and Partitioning are the two steps required to build mini-batches
-   Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
-   Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
-   You have to tune a momentum hyperparameter β and a learning rate α.


## Hyperparameter tuning

### Tuning process

Here are some tips for how to systematically organize your hyperparameter tuning process.

The hyperparameters we like to tune (from must important to Andrew to less important):
-  Learning rate.
-  Momentum beta.
-  Mini-batch size.
-  No. of hidden units.
-  No. of layers.
-  Learning rate decay.
-  Regularization lambda.
-  Activation functions.
-  Adam beta1 & beta2 . # almost never tuned

- In earlier generations of machine learning algorithms, if you had two hyperparameters, it was common practice to sample the points in a grid and systematically explore these values. This practice works okay when the number of hyperparameters was relatively small.

In deep learning, we will just try random values. In practice it's just hard to know in advance which hyperparameters turn out to be the really important hyperparameters for your application and sampling at random rather than in the grid shows that you are more richly exploring set of possible values for the most important hyperparameters, whatever they turn out to be.

- Another common practice is to use a coarse to fine sampling scheme : When you find some hyperparameters values that give you a better performance -> zoom into a smaller region around these values and sample more densely within this space.

- All this methods can be automated.

### Using an appropriate scale to pick hyperparameters

Sampling at random, over the range of hyperparameters, can allow you to search over the space of hyperparameters more efficiently. But it turns out that sampling at random doesn't mean sampling uniformly at random. Instead, it's important to pick the appropriate scale on which to explore the hyperparamaters.

When we looking for the best number of layer or number of hidden units, if we think that a range [2 - 4] for the first and a range of [50, 100] for the second is a good idea, then sampling uniformly at random along these ranges might be reasonable. 

For other hyperparameters,  like alpha, which is very small it's not a good idea :

![hyper_search](https://i.ibb.co/fMVnHDf/image.png)

It's better to search for the right ones using the logarithmic scale rather then in linear scale:
Calculate: 

    a_log = log(a) # e.g. a = 0.0001 then a_log = -4

and : 

    b_log = log(b) # e.g. b = 1 then b_log = 0

Then:

    r = (a_log - b_log) * np.random.rand() + b_log
    # In the example the range would be from [-4, 0] because rand range [0,1)
    result = 10^r

It uniformly samples values in log scale from [a,b] and that prevent us to ignore all the values between 0.0001 and 0.01 in our example.

If we want to use the last method on exploring on the "momentum beta":
- Beta best range is from 0.9 to 0.999.
- You should search for 1 - beta in range 0.001 to 0.1 (1 - 0.9 and 1 - 0.999) and the use a= 0.001 and b = 0.1 . Then:

    a_log = -3
    b_log = -1
    r = (a_log - b_log) * np.random.rand() + b_log
    beta = 1 - 10^r
    # because 1 - beta = 10^r

### Hyperparameters tuning in practice: Pandas vs. Caviar

In deep learning, intuitions about hyperparameter settings from one application area may or may not transfer to a different one.

Andrew recommend maybe just retesting or reevaluating your hyperparameters at least once every several months.

In terms of how people go about searching for hyperparameters, they are two major different ways in which people go about it :

One way is if you babysit one model (if you don't have much computational resources):
- Day 0 you might initialize your parameter as random and then start training.
- Then you watch your learning curve gradually decrease over the day.
- And each day you nudge your parameters a little during training.
This is the panda approach.

Another way if you have enough computational resources, you can run some models in parallel and at the end of the day(s) you check the results.
This is the Caviar approach.

The names are chosen because of panda (and how they have on single baby and make sure it's ok) vs caviar (and fish).

##  Batch Normalization

### Normalizing activations in a network

Now, it turns out that there's one other technique can make your neural network much more robust to the choice of hyperparameters. It doesn't work for all neural networks, but when it does, it can make the hyperparameter search much easier and also make training go much faster : that is batch normalization.

It's one of the most important ideas in the rise of DL and it was created by two researchers, Sergey Ioffe and Christian Szegedy.

Remember that before, we normalized input by subtracting the mean and dividing by variance. This helped a lot for the shape (long to round) of the cost function and for reaching the minimum point faster.

If this work well for the input data, can we do the same on each activation function of each hidden layer l to make training on the layer l+1 more efficient ? That is what batch normalization about.

Note that there are some debates in the deep learning literature about whether you should normalize values before the activation function Z[l] or after applying the activation function A[l] . In practice, normalizing Z[l] is done much more often so let's show we do this :

- Given Z[l] = [z(1), ..., z(m)] , i = 1 to m (for each input)
- Compute mean = 1/m * sum(z[i])
- Compute variance = 1/m * sum((z[i] - mean)^2)
- Then Z_norm[i] = (z[i] - mean) / np.sqrt(variance + epsilon) (add epsilon for numerical stability if variance =0)

Here we are forcing the inputs to a distribution with zero mean and variance of 1.

- Then Z_tilde[i] = gamma * Z_norm[i] + beta

To make inputs belong to other distribution (with other mean and variance). gamma and beta are learn-able parameters of the model. whereas previously you were using these values z1, z2, and so on, you would now use z tilde i, Instead of zi for the later computations in your neural network.

Note: if gamma = sqrt(variance + epsilon) and beta = mean then Z_tilde[i] = z[i]

### Fitting Batch Norm into a neural network

![adding bn to nn](https://i.ibb.co/VNY32G3/image.png)

Our NN parameters will be:
- W[1] , b[1] , ..., W[L] , b[L] , beta[1] , gamma[1] , ..., beta[L] , gamma[L]
beta[1] , gamma[1] , ..., beta[L] , gamma[L] are updated using any optimization algorithms (like GD, RMSprop, Adam).

Note that Batch normalization is usually applied with mini-batches.

- If we are using batch normalization parameters b[1] , ..., b[L] doesn't count because they will be eliminated after mean subtraction step (because taking the mean of a constant b[l] will eliminate the b[l]). So the parameters will be W[l] , beta[l] , and alpha[l] .

Here the implementation of gradient descent with batch normalization :

![implementation](https://i.ibb.co/wyNFzQn/image.png)

### Why does Batch Norm work?

One intuition behind why batch norm works is, just like how by normalizing the input features, the X's, to mean zero and variance one, that can speed up learning, this is doing a similar thing, but further values in your hidden units and not just for your input there.
Now, this is just a partial picture for what batch norm is doing.


A second reason why batch norm works is it makes weights, later or deeper in your network, more robust to changes then weights in earlier layers of the neural network.

- the batch normalization reduces the amount that the distribution of these hidden unit values shifts around. (black cat vs color cat classification). it limits the amount to which updating the parameters in the earlier layers can affect the distribution of values that
the later layer now sees and therefore has to learn on.

So batch norm reduces the problem of the input values changing, it really causes these values to become more stable so that the later layers of the neural network has more firm ground to stand on and even though the input distribution changes a bit, it changes less.

Batch normalization does some regularization:
- Each mini batch is scaled by the mean/variance computed of that mini-batch.
This adds some noise to the values Z[l] within that mini batch. So similar to dropout it adds some noise to each hidden layer's activations.
- This has a slight regularization effect.
- Using bigger size of the mini-batch you are reducing noise and therefore regularization effect.
- Don't rely on batch normalization as a regularization. It's intended for normalization of hidden units, activations and therefore speeding up learning. For regularization use other regularization techniques (L2 or dropout).

### Batch Norm at test time

Batch norm processes your data one mini batch at a time, but the test time you might need to process the examples one at a time. The mean and the variance of one example won't make sense. Let's see how we can adapt the network to do that.

We have to compute an estimated value of mean and variance to use it in testing time. for that we can use the weighted average across the mini-batches. and at the end, we will use the estimated values of the mean and variance to test. This method is also sometimes called by the fancy name of "Running average".

In practice most often you will use a deep learning framework and it will contain some default implementation of doing such a thing.

## Multi-class classification

### Softmax Regression

The classification examples we've talked about have used binary classification, where we had two possible labels, 0 or 1. There's a generalization of logistic regression called Softmax regression fo the case where we have multiple possible classes.

Let's take an example where we try to classify cats (class 1), dogs (class 2), baby chicks (class 3) and none of the above (class 0).

The notation we are going to use is :
- C = no. of classes
- Range of classes is (0, ..., C-1)
- In output layer Ny = C.

Each of C values in the output layer will contain a probability of the example to belong to each of the classes.

Here are the equations of the softmax activation function:

    t = e^(Z[L]) # shape(C, m)
    A[L] = e^(Z[L]) / sum(t) # shape(C, m), sum(t) - sum of t's for each example (shape (1, m))  

### Training a softmax classifier

The name softmax comes from contrasting it to what's called a hard max which would have taken the vector Z and just put a 1 in the position of the biggest element  of Z and then 0s everywhere else. Whereas in contrast, a softmax is a more gentle mapping from Z to these probabilities.

Softmax regression or generalizes the logistic activation function to C classes rather than just two classes. And it turns out that if C = 2, then softmax with C = 2 essentially reduces to logistic regression. The proof is that if C = 2 and if you apply softmax then the result should be the same as if you applied  Logistic regression.

The loss function used with softmax is :

    L(y, y_hat) = - sum(y[j] * log(y_hat[j])) # j = 0 to C-1

The cost function used with softmax:

    J(w[1], b[1], ...) = - 1 / m * (sum(L(y[i], y_hat[i]))) # i = 0 to m

Back propagation with softmax:

    dZ[L] = Y_hat - Y

The derivative of softmax is:

    Y_hat * (1 - Y_hat)

## Introduction to programming frameworks


### Deep learning frameworks

Today, there are many deep learning frameworks that makes it easy for you to implement neural networks, and here are some of the leading ones. 

- Caffe/ Caffe2
- CNTK
- DL4j
- Keras
- Lasagne
- mxnet
- PaddlePaddle
- TensorFlow
- Theano
- Torch/Pytorch

How to choose deep learning framework:
- Ease of programming (development and deployment)
- Running speed
- Truly open (open source with good governance)

Note that frameworks can not only shorten your coding time but sometimes also perform optimizations that speed up your code.

### TensorFlow

this video is shows the basic structure of a TensorFlow program. Let's see how to implement the cost function J :

    J(w) = w^2 - 10w + 25

Here we recognizee the `(w-5)^2` form of the equation, thus W=5 is the minimum we except.

In TensorFlow you implement only the forward propagation and TensorFlow will do the backpropagation by itself.

In TensorFlow a placeholder is a variable you can assign a value to later.

If you are using a mini-batch training you should change the feed_dict={x: coefficients} to the current mini-batch
data.

In deep learning frameworks there are a lot of things that you can do with one line of code like changing the optimizer.

### Notebook notes

Writing and running programs in TensorFlow has the following steps:

1.  Create Tensors (variables) that are not yet executed/evaluated.
2.  Write operations between those Tensors.
3.  Initialize your Tensors.
4.  Create a Session.
5.  Run the Session. This will run the operations you'd written above.

A placeholder is simply a variable that you will assign data to only later, when running the session. We say that you **feed data** to these placeholders when running the session.

For computing a function like sigmoid :
2.  Create placeholders
3.  Specify the computation graph corresponding to operations you want to compute
4.  Create the session
5.  Run the session, using a feed dictionary if necessary to specify placeholder variables' values.

**What you should remember**:

-   Tensorflow is a programming framework used in deep learning
-   The two main object classes in tensorflow are Tensors and Operators.
-   When you code in tensorflow you have to take the following steps:
    -   Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
    -   Create a session
    -   Initialize the session
    -   Run the session to execute the graph
-   You can execute the graph multiple times as you've seen in model()
-   The backpropagation and optimization is automatically done when running the session on the "optimizer" object.
