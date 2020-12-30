# Convolutional Neural Networks

### Computer Vision

Computer vision is one of the areas that's been advancing rapidly thanks to deep learning.

We use daily computer vision + deep learning, on our phone (unlocking), with some apps etc. 

Computer vision deep leaning techniques are always evolving making a new architectures which can help us in other areas other than computer vision.

In this course, we are going to see several problems:
- Image classification
- Object detection
- Transfer learning.

Some of the problem we can have in computer vision is the size of the images. If the image is too large, then our NN (with fully connected layers) will have too many weight to compute and with that much weight, it's very hard to not overfit, or to run correctly the algorithm with enough memory or computational requirements.

As we want to use large images, we need to implement the convolution operation, which is one of the fundamental building blocks of convolutional neural networks.

### Edge Detection Example

Let's see how the convolution operation work with one example of edge detection.

We already said that the early layers of the neural network might detect edges and then some later layers might detect part of objects and then even later layers may detect complete objects like people's faces in this case.

Let's see the first part : detecting edges.

- We have a `6 x 6` gray scale image matrix.
- In order to detect the edges, we are going to construct a ` 3 x 3` matrix that we will call the **filter** or **kernel** .
- Then, we are going to **convolve** the first matrix with the second one.  Note that the convolution part is symbolized by `*` (So this is not element-wise multiplication. 

- The output will be a `4 x 4` matrix (image).

To compute the convolution we take the filter and apply it to same region in term of pixel (and move it to cover all the original matrix) , do a element-wise multiplication and addition. Just like this :

![convolution](https://i.ibb.co/vZsqFhS/image.png)
If you make the convolution operation in TensorFlow you will find the function tf.nn.conv2d . In keras you will find Conv2d function.

Why or how this is detecting the edges ? 

![conv](https://i.ibb.co/JjzzSvZ/image.png)

In this example, this bright region in the middle is just the output images way of saying that it looks like there is a strong vertical edge right down the middle of the image.

One intuition is that a vertical edge is where there are  bright pixels on the left and darker pixels on the right.
So that the the vertical edge detection filter will find a 3x3 place in an image where there are a bright region followed by a dark region.


### More Edge Detection

We saw vertical edges detection, for horizontal one we can assume that the filter can be like :

![enter image description here](https://i.ibb.co/GtTxj4Z/image.png)
Note that there are other edge detection filters like the Sobel and Scharr filter but what we want is that we don't need to hand craft these numbers, we can treat them as weights and then learn them. It can learn horizontal, vertical, angled, or any edge type automatically rather than getting them by hand.

### Padding

In order to build deep neural networks one modification to the basic convolutional operation that you need to really use is **padding**.

if you have a `n x n` image to convolve with an `f x f` filter, then the dimension of the output will be `(n-f+1 x n-f+1)`

 There re two downside to that :
 - The convolution operation shrinks the matrix and we don't want our image to shrink every time we detect edges or to set other features on it.
 - the edges pixels are used less than other pixels in an image so we ending up throwing away a lot of information that are in the edges.

to fix these two problems, one solution is to **pad** the image, for example, our `6 x 6 ` is now a `8 x 8` matrix and the output after convolution operation will be a `6 x 6` matrix.

By convention when you pad, you padded with zeros and if p is the padding amounts.
The general rule now, if a matrix `n x n` is convolved with `f x f` filter/kernel and padding p give us `n+2p-f+1,n+2p-f+1` matrix.

Note that "valid convolution" is a convolution operation without padding and  "same convolution" is a convolution with a pad so that output size is the same as the input size. Its given by the equation:

    P = (f-1) / 2

In computer vision f is usually odd. Some of the reasons is that its have a center value and that we can refer to it easily.
 
To specify the padding for your convolution operation, you can either specify the value for p or you can just say that this is a valid convolution, which means p equals zero or you can say this is a same convolution, which means pad as much as you need to make sure the output has same dimension as the input.

### Strided Convolutions

When we are making the convolution operation we used `S` to tell us the number of pixels we will jump when we are convolving filter/kernel. The last examples we described S was 1.

Now the general rule are:

- if a matrix `n x n` is convolved with `f x f` filter/kernel and padding p and stride s it give us `(n+2p-f)/s + 1,(n+2pf)/s + 1` matrix.
- In case `(n+2p-f)/s + 1` is fraction we can take floor of this value.
- The filter matrix must be fully contained within the image or the image plus the padding otherwise, we don't compute it. 

In math textbooks the convolution operation is flipping the filter before using it. What we were doing is called cross-correlation operation but the state of art of deep learning is using this as convolution operation.

Same convolutions equation become :

    p = (n*s - n + f - s) / 2
    When s = 1 ==> P = (f-1) / 2

The flipping in the definition of convolution causes convolution operator to enjoy the property "associativity" in mathematics. This is nice for some signal processing applications but for deep neural networks it really doesn't matter.

### Convolutions Over Volume

let's see how you can implement convolutions over, not just 2D images, but over three dimensional volumes.

We will convolve an image of height, width, # of channels with a filter of a height, width, same # of channels. Note that the image number channels and the filter number of channels must be the same:

![conin3d](https://i.ibb.co/c8wP6nr/image.png)


- The output is a 2D array.
- If only one of the 3 channel is non-zeros then we are "filtering" only on that channel (in the example where only red is != 0, then we are looking for edges in the "red matrix".

If we want to detect several edges and thus use several filters, then the solution is to make the convolution operation with all the filters and then stack the result together just like this :


![multiple filters](https://i.ibb.co/WxqjLfk/image.png)


The output will then have a number of channels equal to the number of filters you are detecting.

### One Layer of a Convolutional Network

Let's take an example of how to compute one layer of a convolutional neural network.

![one_layer_conv](https://i.ibb.co/CbCXQKZ/image.png)

- Our filter is our weights, so applying a convolution operation on our input value is computing ` W[l] * a[l-1]`(the purple 2D matrix in the example).  To this we add the bias b[l], that would be our Z[l]. What we do next is applying relu function on the result  Z[l] and finally, the computation give us A[l].

Here are some notations we will use.
If layer l is a conv layer, so:

    f[l] = filter size
    p[l] = padding # Default is zero
    s[l] = stride
    nc[l] = number of filters
    Input: n[l-1] x n[l-1] x nc[l-1] Or nH[l-1] x nW[l-1] x nc[l-1]
    Output: n[l] x n[l] x nc[l] Or nH[l] x nW[l] x nc[l]
    Where n[l] = (n[l-1] + 2p[l] - f[l] / s[l]) + 1
    Each filter is: f[l] x f[l] x nc[l-1]
    Activations: a[l] is nH[l] x nW[l] x nc[l]
			 A[l] is m x nH[l] x nW[l] x nc[l] # In batch or minbatch training
    Weights: f[l] * f[l] * nc[l-1] * nc[l]
    bias: (1, 1, 1, nc[l])


### Simple Convolutional Network Example

Here is the example :

![convnet](https://i.ibb.co/FhjR5N5/image.png)

Notice that when sride is 2, the shrinking goes much faster.
The matrix hight and width are getting smaller and smaller while the matrix depth are getting larger.

So there is our example, let see next the orther type of layer in CNN:

- Pooling (POOL)
- Fully connected (FC)


### Pooling Layers

Other than convolutional layers, ConvNets often also use pooling layers to reduce the size of the representation, to speed the computation, as well as make some of the features it detects a bit more robust.


There are two type of Pooling layers:
- Average pooling
- Max pooling

Let's take a look at max pooling:

![maxpooling](https://i.ibb.co/XpPytwr/image.png)


The intuition behind that is : if the feature is detected anywhere in this filter then keep a high number. But the main reason why people are using pooling because its works well in practice and reduce computations.

Note that in the pooling layer, there are no parameters to learn, we just have to fix `f` and `s`.  In most cases, we use `f=s=2`. In the case where `f=s` then this result at shrinking the input.

The average pooling in other hand, take the average of the values of the matrix instead of the max value. In practice, Max pooling is more often used.

Also, notice that we rarely use a padding in pooling.

So, to summary, these parameters will be hyperparameters, and therefor we will have :
- f: the filter size
- s :stride
- the padding if applied
- if we choose max pooling or average pooling

### CNN Example


Here is one complete example of a CNN : 

![cnn](https://i.ibb.co/gw96X8M/image.png)
This is inspired from the LeNet-5 architecture created by Yann LeCun.

Here, one layer is only the layers that have weights so in CNN, one layer equal to conv + Pool.

We can see that FC layers have the most parameters (in  FC3 : 400*120 + 120 = 48120). 

To choose the best hyperparameters, one must read the literature. 


### Why Convolutions?

there are two main advantages of convolutional layers over just using fully connected layers :

- Parameter sharing
A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.
- sparsity of connections
In each layer, each output value depends only on a small number of inputs which makes it translation invariance.

Putting it together :

![cnn_complete_training](https://i.ibb.co/zb71kh6/image.png)

Notebook notes:

The main benefits of padding are the following:

-   It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.
    
-   It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

In TensorFlow, there are built-in functions that implement the convolution steps for you.

-   **tf.nn.conv2d(X,W, strides = [1,s,s,1], padding = 'SAME'):** given an input XX and a group of filters WW, this function convolves WW's filters on X. The third parameter ([1,s,s,1]) represents the strides for each dimension of the input (m, n_H_prev, n_W_prev, n_C_prev). Normally, you'll choose a stride of 1 for the number of examples (the first value) and for the channels (the fourth value), which is why we wrote the value as `[1,s,s,1]`. You can read the full documentation on [conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d).
    
-   **tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'):** given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. For max pooling, we usually operate on a single example at a time and a single channel at a time. So the first and fourth value in `[1,f,f,1]` are both 1. You can read the full documentation on [max_pool](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool).
    
-   **tf.nn.relu(Z):** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu).
    
-   **tf.contrib.layers.flatten(P)**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.
    
    -   If a tensor P has the shape (m,h,w,c), where m is the number of examples (the batch size), it returns a flattened tensor with shape (batch_size, k), where k=h×w×ck=h×w×c. "k" equals the product of all the dimension sizes other than the first dimension.
    -   For example, given a tensor with dimensions [100,2,3,4], it flattens the tensor to be of shape [100, 24], where 24 = 2 _3_ 4. You can read the full documentation on [flatten](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten).
-   **tf.contrib.layers.fully_connected(F, num_outputs):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [full_connected](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected).
    

In the last function above (`tf.contrib.layers.fully_connected`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

##### Window, kernel, filter[](https://godrdpxw.labs.coursera.org/notebooks/week1/Convolution_model_Application_v1a.ipynb#Window,-kernel,-filter)

The words "window", "kernel", and "filter" are used to refer to the same thing. This is why the parameter `ksize` refers to "kernel size", and we use `(f,f)` to refer to the filter size. Both "kernel" and "filter" refer to the "window."

## Deep convolutional models: case studies

### why look at case study

We learned about the basic building blocks such as convolutional layers, proving layers and fully connected layers of conv nets. It turns out a lot of the past few years of computer vision research has been on how to put together these basic building blocks to form effective CNN.

A good way to get intuition on how to build conv nets is to read or to see other examples of effective conv nets.

After the next few videos, we'll be able to read some of the research papers from the theater computer vision.

We will see these classic networks:

- LeNet-5
- AlexNet
- VGG

Then, we will see **Resnet**, a deeper CNN with 152 layers and which won the ImageNet competition and after that **Inception** made by Google.

Reading and trying the mentioned models can boost you and give you a lot of ideas to solve your task.


### Classic Networks

In this section, we are going to talk about the classic network mentioned before. 

#### LeNet-5

It's from a paper by "LeCun et al." in 1998 "Gradient-based learning applied to document recognition".

The goal of the NN is to recognize handwritten digits in a `32 x 32 x 1` gray image.

When the paper was published :
- people used average pooling
- people didn't use padding (only valid convolutions)
- The last layers was not softmax, the function back then is now useless today.

 ![LeNet-5](https://i.ibb.co/qrMkBhj/image.png)
 To summarize, LeNet-5 architecture used 5 layers, with 4 hidden layers (CONV-POOL (2x), FC, FC). This type of arrangement of layers is quite common. The model was small (60K parameters) vs 10M to 1000M parameters today.

The activation function used in the paper was Sigmoid and Tanh. Modern implementation uses RELU in most of the cases. 

If you read the paper, focus on part II and part III and after that on the experimentation section. You can find the link [here](https://ieeexplore.ieee.org/document/726791?reload=true).

#### AlexNet


The model is named after Alex Krizhevsky, the first author of the paper. The other author's were Ilya Sutskever and Geoffrey Hinton.

It was train on the ImageNet data (challenge with a goal to classify images into 1000 different classes) and the paper was published in 2012 "ImageNet classification with deep convolutional neural networks".

  ![AlexNet](https://i.ibb.co/K0csD2d/image.png)

It's similar to LeNet-5 but much bigger, the AlexNet have 160M parameters and it used the RELU activation function that was made the model better then LeNet-5 (that with the training length)  

To summarize, AlexNet architecture used 8 layers, with 7 hidden layers (CONV-POOL (2x), CONV, CONV, CONV-POOL, FC, FC).

Back then, GPU was slower, so multiple GPUs was used.

Researchers proved that Local Response normalization doesn't help much so for now don't bother yourself for
understanding or implementing it.

This paper convinced the computer vision researchers that deep learning is so important. The paper can be find [here](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

### VGG

This is a modification of the AlexNet, published  by Simonyan & Zisserman in 2015 called " Very deep convolutional networks for large-scale image recognition". 

Note that this paper is the most easy ones to read.

 It was based on the idea : instead of having a lot of hyperparameters let's use a much simpler network where we focus on just having :

- CONV by `3 x 3` filter and `s=1` , same 
- Max-POOL by `2x2`, s=2

Here is the architecture :

![vgg16](https://i.ibb.co/q0vsVq0/image.png)


 The 16 in the name refer to the fact that there are 16 layers that have weights.
 
 - Pooling was the only one who is responsible for shrinking the dimensions.
 - The model is pretty large (138M parameters) The most of them are in the FC layers.
-  The simplicity of the VGG-16 architecture made it quite appealing and the relative uniformity of this architecture made it quite attractive to researchers.
- The VGG-19 is a bigger version but most people uses the VGG-16 instead of the
VGG-19 because it does the same.

### ResNets



Very deep neural networks are difficult to train because of vanishing and exploding gradient types of problems. In this video, you'll learn about skip connections which allows you to take the activation from one layer and suddenly feed it to another layer even much deeper in the neural network. And using that, you'll build ResNet which enables you to train very, very deep networks.

ResNets are built out of some Residual blocks :

![](https://i.ibb.co/Y7F5scJ/image.png)

What we do here is adding `a^l` to `z^l+2`, so we are skiping a connexion and we added a short cut.

By doing so, the authors of this methods find out that we can train a very deep NN. 

These networks can go deeper without hurting the performance. In the normal NN - Plain networks - the theory tell us that if we go deeper we will get a better solution to our problem, but because of the vanishing and exploding gradients problems the performance of the network suffers as it goes deeper. Thanks to Residual Network we can go deeper as we want now.

### Why ResNets Work

Let's take one example that explained why ResNet works:

We already said that adding layers to your NN might hurt its performances. This statement is less true when using ResNet. Let's take a NN, make it deeper and suppose we are using the ReLu :

![](https://i.ibb.co/c1mbkFM/image.png)

Then if we are using L2 regularization for example, W[l+2] will be zero. Lets say that b[l+2] will be zero too, then a[l+2] = g( a[l] ) = a[l] with no negative values.
This show that **identity function is easy for a residual block to learn.** And that why it can train deeper NNs. 

Note that the two layers we added doesn't hurt the performance of big NN we made.
if all of these heading units learned something useful then maybe you can do even better than learning the identity function. And what goes wrong in very deep plain nets in very deep network without this residual of the skip connections is that when you make the network deeper and deeper, it's actually very difficult for it to choose parameters that learn even the identity function which is why a lot of layers end up making your result worse rather than making your result better.


Hint: dimensions of z[l+2] and a[l] have to be the same in ResNets. In case they have different dimensions what we put a matrix parameters (Which can be learned or fixed)

    a[l+2] = g( z[l+2] + ws * a[l] ) # The added Ws should make the dimensions equal

- ws also can be a zero padding.

Using a skip-connection helps the gradient to backpropagate and thus helps you to train deeper networks.

I think that when Andrew says its easy to learn the identity for the Res Blocks, it's in the case where the perfs are "worst" with additional block, and just computing g(a) makes the model the same as without the block.

To transform a plain NN to a ResNet :

![](https://i.ibb.co/gFBpMn7/image.png)

### Networks in Networks and 1x1 Convolutions

In terms of designing content architectures, one of the ideas that really helps is using a one by one convolution. Now, you might be wondering, what does a one by one convolution do? Isn't just multiplying by a number ? Let's take two examples :


![](https://i.ibb.co/6rP5yfm/image.png)

In the first example, we end up just multiplying the `6x6` matrix by 2. - Not very useful.
With the second example thus, a `6 x 6x 32` matrix convolved by a ` 1 x 1 x 32` filter makes much more sense.

This will look at each of the 36 different positions here, and it will take the element wise product between 32 numbers on the left and 32 numbers in the filter and then apply a ReLU non-linearity to it after that.
So one way to think about the one by one convolution is that, it is basically having a fully connected neuron network. The NN will take a slice of our input as the input and applying the filter (weight) then ReLu to finally output our matrix with #filters as the third dimension. 

Note that a ` 1 x 1` convolution is also called a Network in Network.

A 1 x 1 convolution is useful when we want to shrink the number of channels. We also call this feature transformation. We will later see that by shrinking it we can save a lot of computations.

It is also useful if we have specified the number of 1 x 1 Conv filters to be the same as the input number of channels then the output will contain the same number of channels. Then the 1 x 1 Conv will act like a non linearity and will learn non linearity operator.

Note : the original paper (Lin 2013 - Network in Network) states "Convolution layers take **inner product** of the linear filter and the underlying receptive field followed by a nonlinear activation function at every local portion of the input. ". That means for layer l you need to calculate (in terms of dimensions)

    (1 x n_c) x (n_c x 1) = 1 x 1

 

### Inception Network Motivation

When designing a ConvNet, one have to pick the filter size, if pooling must be used and so many other choice to make. The Inception Network tells us **why not using them all ?**  This will be more complicated network architecture but it also works very well!

Let's see it :

 ![](https://i.ibb.co/dJ4JdqT/image.png)

Here what we just did is applying all the convs and pools layers that we might want and just stack them together. So the input to the inception module are `28 x 28 x 192` and the output are `28 x 28 x 256`.

We will let the NN decide wich it want to use the must.

The problem of this method is the computational cost. If we have just focused on a 5 x 5 Conv that we have done in the last example : There are 32 same filters of 5 x 5, and the input are `28 x 28 x 192`. the output should be `28 x 28 x 32`.

The total number of multiplications needed here are:

Number of outputs * Filter size * Filter size * Input dimension which equals: 28 * 28 * 32 * 5 * 5 * 192 = 120 M

120 Mil multiply operation still a problem in the modern day computers.


Using a 1 x 1 convolution we can reduce 120 mil to just 12 mil. Lets see how.

![](https://i.ibb.co/NZ21TYD/image.png)

A 1 x 1 Conv here is called Bottleneck `BN`

So to summarize, if you are building a layer of a neural network and you don't want to have to decide, do you want a 1 by 1, or 3 by 3, or 5 by 5, or pooling layer, the inception module let's you say let's do them all, and let's concatenate the results.

And then we run to the problem of computational cost. And what you saw here was how using a 1 by 1 convolution, you can create this bottleneck layer thereby reducing the computational cost significantly.

Now you might be wondering, does shrinking down the representation size so dramatically, does it hurt the performance of your neural network?

It turns out that so long as you implement this bottleneck layer so that within reason, you can shrink down the representation size significantly, and it doesn't seem to hurt the performance, but saves you a lot of computation.

### Inception Network

We've already seen all the basic building blocks of the Inception network.

The inception network consist of concatenated blocks of the Inception module.

The name inception was taken from a meme image which was taken from Inception movie

the inception network is largely the inception module repeated a bunch of times throughout the network. Since the development of the original inception module, the author and others have built on it and come up with other versions as well.

So there are research papers on newer versions of the inception algorithm. And you sometimes see people use some of these later versions as well in their work, like inception v2, inception v3, inception v4

## Practical advices for using ConvNets

### Using Open-Source Implementation

We've now learned about several highly effective neural network and ConvNet architectures. Here some practical advice on how to use them, first starting with using open source implementations.

It turns out that a lot of these neural networks are difficult to replicate because a lot of details about tuning of the hyperparameters such as learning decay and other things and it's sometime difficult even for the best PhD student to replicate someone else's polished work just from reading their paper.

Fortunately, a lot of deep learning researchers routinely open source their work on the Internet, such as on GitHub and we should do the same.

If you see a research paper and you want to build over it, the first thing you should do is to look for an open source implementation for this paper.

### Transfer Learning

Sometimes networks take a long time to train and someone else might have used multiple GPUs and a very large dataset to pretrain some of these networks. And that allows you to do transfer learning using these networks.

So if you are using a specific NN architecture that has been trained before, you can use this pretrained parameters/weight instead of random initialization to solve your problem.

Depend on the amont of data we have, there are different recommendation for how to do this. Let's take an example with a cat classifier.

If we don't have a lot of data :

- The cat classification problem contains 3 classes Tigger, Misty and neither.
- Andrew recommends to go online and download a good NN with its weights, remove the softmax activation layer and put your own one and make the network learn only the new layer while other layer weights are fixed/frozen.
- Frameworks have options to make the parameters frozen in some layers using `trainable = 0 or freeze = 0`
- One of the tricks that can speed up your training, is to run the pretrained NN without final softmax layer and get an intermediate representation of your images and save them to disk. And then use these representation to a shallow NN network. This can save you the time needed to run an image through all the layers. Its like converting your images into vectors.

If we have a lot of data :

- One thing you can do is to freeze few layers from the beginning of the pretrained network and learn the other weights in the network.
- Some other idea is to throw away the layers that aren't frozen and put your own layers there.

If we **really** have a lot of data :

- You can fine tune all the layers in your pretrained network but **don't random initialize** the parameters, leave the learned parameters as it is and learn from there.

### Data Augmentation

If data is increased, your deep NN will perform better. Data augmentation is one of the techniques that deep learning uses to increase the performance of deep NN.

This is true is you're using transfer learning as well.

The data augmentation methods used in computer vision includes :
- mirroring
- random cropping :
	- if we wrong crop, that can be a problem.
	- make sure tour crops is big enough to not have this problem.
- rotation, shearing, local warpinp (used a bit less)
- Color Shifting
	- adding distortions to R, G and B (more green, more blue etc) it will be different for the computer but the identity of the content stay the same. That make the algorithm more robust to identify cat (for example) even if the colors are changing.
	- There are an algorithm which is called **PCA color augmentation** that decides the shifts needed automatically.

How we should implement distortions during training? That depends on how many data we have. 

You might have your training data stored in a hard disk, with a small training set, you can do almost anything and you'll be okay.

But for very last training set, this is how people will often implement it: 

- you might use a different CPU thread to make you a distorted mini batches while you are training your NN.

Note that Data Augmentation has also some hyperparameters. A good place to start is to find an open source data augmentation implementation and then use it or fine tune these hyperparameters.

### State of Computer Vision

Deep learning has been successfully applied to many problems. There are few things unique to DL in computer vision. Here are some observations about deep learning for computer vision.


![](https://i.ibb.co/cJqWsP8/image.png)

Depend on our problem, we can have a lot or just a little amount of data. If we don't have that much data people tend to try more hand engineering for the problem "Hacks". Like choosing a more complex NN architecture. If we have more data, then researchers are tend to use simpler algorithms and less hand engineering.

Because we haven't got that much data in a lot of computer vision problems, it relies a lot on hand engineering. One other thing to try here is Transfer Learning. 

And so that's another set of techniques that's used a lot for when you have relatively little data. If you look at the computer vision literature, and look at the sort of ideas out there, you also find that people are really enthusiastic. They're really into doing well on standardized benchmark data sets and on winning competitions.

So here some tips for doing well on benchmarks/winning competitions:

- Ensembling.
	- Train several networks independently and average their outputs. (sometime make you gain 2% but  this will slow down your production by the number of the ensembles. So this is used in compt. /bench. but never in production. Also it takes more memory as it saves all the models in the memory.
	- To do that : after you decide the best architecture for your problem, initialize some of that randomly and train them independently.

- Multi-crop at test time.
	- Run classifier on multiple versions of test versions and average results.
	- There is a technique called 10 crops that uses this.This can give you a better result in the production.

- Use open source code
	- Use architectures of networks published in the literature.
	- Use open source implementations if possible.
	- Use pretrained models and fine-tune on your dataset.

### Notebook notes

#### 1 - The problem of very deep neural networks

Last week, you built your first convolutional neural network. In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers.

* The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the shallower layers, closer to the input) to very complex features (at the deeper layers, closer to the output). 
* However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent prohibitively slow. 
* More specifically, during gradient descent, as you backprop from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode" to take very large values). 
* During training, you might therefore see the magnitude (or norm) of the gradient for the shallower layers decrease to zero very rapidly as training proceeds: 

<img src="images/vanishing_grad_kiank.png" style="width:450px;height:220px;">
<caption><center> <u> <font color='purple'> **Figure 1** </u><font color='purple'>  : **Vanishing gradient** <br> The speed of learning decreases very rapidly for the shallower layers as the network trains </center></caption>

You are now going to solve this problem by building a Residual Network!

#### 2 - Building a Residual Network

In ResNets, a "shortcut" or a "skip connection" allows the model to skip layers:  

<img src="images/skip_connection_kiank.png" style="width:650px;height:200px;">
<caption><center> <u> <font color='purple'> **Figure 2** </u><font color='purple'>  : A ResNet block showing a **skip-connection** <br> </center></caption>

The image on the left shows the "main path" through the network. The image on the right adds a shortcut to the main path. By stacking these ResNet blocks on top of each other, you can form a very deep network. 

We also saw in lecture that having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that you can stack on additional ResNet blocks with little risk of harming training set performance.  
    
(There is also some evidence that the ease of learning an identity function accounts for ResNets' remarkable performance even more so than skip connections helping with vanishing gradients).

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. You are going to implement both of them: the "identity block" and the "convolutional block."## 2 - Building a Residual Network

In ResNets, a "shortcut" or a "skip connection" allows the model to skip layers:  

<img src="images/skip_connection_kiank.png" style="width:650px;height:200px;">
<caption><center> <u> <font color='purple'> **Figure 2** </u><font color='purple'>  : A ResNet block showing a **skip-connection** <br> </center></caption>

The image on the left shows the "main path" through the network. The image on the right adds a shortcut to the main path. By stacking these ResNet blocks on top of each other, you can form a very deep network. 

We also saw in lecture that having ResNet blocks with the shortcut also makes it very easy for one of the blocks to learn an identity function. This means that you can stack on additional ResNet blocks with little risk of harming training set performance.  
    
(There is also some evidence that the ease of learning an identity function accounts for ResNets' remarkable performance even more so than skip connections helping with vanishing gradients).

Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different. You are going to implement both of them: the "identity block" and the "convolutional block."

##### 2.1 - The identity block

The identity block is the standard block used in ResNets, and corresponds to the case where the input activation (say $a^{[l]}$) has the same dimension as the output activation (say $a^{[l+2]}$). To flesh out the different steps of what happens in a ResNet's identity block, here is an alternative diagram showing the individual steps:

<img src="images/idblock2_kiank.png" style="width:650px;height:150px;">
<caption><center> <u> <font color='purple'> **Figure 3** </u><font color='purple'>  : **Identity block.** Skip connection "skips over" 2 layers. </center></caption>

The upper path is the "shortcut path." The lower path is the "main path." In this diagram, we have also made explicit the CONV2D and ReLU steps in each layer. To speed up training we have also added a BatchNorm step. Don't worry about this being complicated to implement--you'll see that BatchNorm is just one line of code in Keras! 

In this exercise, you'll actually implement a slightly more powerful version of this identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers. It looks like this: 

<img src="images/idblock3_kiank.png" style="width:650px;height:150px;">
<caption><center> <u> <font color='purple'> **Figure 4** </u><font color='purple'>  : **Identity block.** Skip connection "skips over" 3 layers.</center></caption>

Here are the individual steps.

First component of main path: 
- The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be `conv_name_base + '2a'`. Use 0 as the seed for the random initialization. 
- The first BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '2a'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Second component of main path:
- The second CONV2D has $F_2$ filters of shape $(f,f)$ and a stride of (1,1). Its padding is "same" and its name should be `conv_name_base + '2b'`. Use 0 as the seed for the random initialization. 
- The second BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '2b'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Third component of main path:
- The third CONV2D has $F_3$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be `conv_name_base + '2c'`. Use 0 as the seed for the random initialization. 
- The third BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '2c'`. 
- Note that there is **no** ReLU activation function in this component. 

Final step: 
- The `X_shortcut` and the output from the 3rd layer `X` are added together.
- **Hint**: The syntax will look something like `Add()([var1,var2])`
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

**Exercise**: Implement the ResNet identity block. We have implemented the first component of the main path. Please read this carefully to make sure you understand what it is doing. You should implement the rest. 
- To implement the Conv2D step: [Conv2D](https://keras.io/layers/convolutional/#conv2d)
- To implement BatchNorm: [BatchNormalization](https://faroit.github.io/keras-docs/1.2.2/layers/normalization/) (axis: Integer, the axis that should be normalized (typically the 'channels' axis))
- For the activation, use:  `Activation('relu')(X)`
- To add the value passed forward by the shortcut: [Add](https://keras.io/layers/merge/#add)


##### 2.2 - The convolutional block

The ResNet "convolutional block" is the second block type. You can use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path: 

<img src="images/convblock_kiank.png" style="width:650px;height:150px;">
<caption><center> <u> <font color='purple'> **Figure 4** </u><font color='purple'>  : **Convolutional block** </center></caption>

* The CONV2D layer in the shortcut path is used to resize the input $x$ to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path. (This plays a similar role as the matrix $W_s$ discussed in lecture.) 
* For example, to reduce the activation dimensions's height and width by a factor of 2, you can use a 1x1 convolution with a stride of 2. 
* The CONV2D layer on the shortcut path does not use any non-linear activation function. Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step. 

The details of the convolutional block are as follows. 

First component of main path:
- The first CONV2D has $F_1$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be `conv_name_base + '2a'`. Use 0 as the `glorot_uniform` seed.
- The first BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '2a'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Second component of main path:
- The second CONV2D has $F_2$ filters of shape (f,f) and a stride of (1,1). Its padding is "same" and it's name should be `conv_name_base + '2b'`.  Use 0 as the `glorot_uniform` seed.
- The second BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '2b'`.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 

Third component of main path:
- The third CONV2D has $F_3$ filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and it's name should be `conv_name_base + '2c'`.  Use 0 as the `glorot_uniform` seed.
- The third BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '2c'`. Note that there is no ReLU activation function in this component. 

Shortcut path:
- The CONV2D has $F_3$ filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be `conv_name_base + '1'`.  Use 0 as the `glorot_uniform` seed.
- The BatchNorm is normalizing the 'channels' axis.  Its name should be `bn_name_base + '1'`. 

Final step: 
- The shortcut and the main path values are added together.
- Then apply the ReLU activation function. This has no name and no hyperparameters. 



#### What you should remember[](https://godrdpxw.labs.coursera.org/notebooks/week2/ResNets/Residual_Networks_v2a.ipynb#What-you-should-remember)

-   Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.
-   The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function.
-   There are two main types of blocks: The identity block and the convolutional block.
-   Very deep Residual Networks are built by stacking these blocks together.

## Object detection

### Object localization 

This week is about object detection. This is an area of CV that is exploding and work much better. 
To do object detection, we need first to do object localization. Let's start with this.

What are localization and detection ?

![](https://i.ibb.co/LCLD1C3/image.png)

The ideas for image classification will be useful for classification with localization and the ideas for localization will then turn out to be useful for detection

We are already familiar with the image classification : input image, pass it to a ConvNet, add  softmax layer with the possible output. If we want to add here a localization, then we can change the NN to output the bounding box of the detected object. 

We need to attached to the end of the new NN four numbers : bx , by , bh , and bw to tell the location of the class in the image. The dataset should contain this four numbers
with the class too. (bx, by) is the midpoint of the bounding box and bh, bw is the height and the width. 

Note that as we need to give position, we also need to precise that the upper left of the image will be (0, 0) and the bottom right will be (1,1).
 
Example:

    Y = [
    Pc # Probability of an object is presented
    bx # Bounding box
    by # Bounding box
    bh # Bounding box
    bw # Bounding box
    c1 # The classes
    c2
    c3
    ...
    ]

so that will give us :

![](https://i.ibb.co/xhry9Rv/image.png)

 ? means we don't care with other values.
 
Finally, the loss function that we can use will be :

    L(y',y) = {
	    (y1'-y1)^2 + (y2'-y2)^2 + ... if y1 = 1
	    (y1'-y1)^2 if y1 = 0
    }

Here we used the square error just for the representation, but in practice we usually use logistic regression loss for pc , log likely hood loss for classes, and squared error for the bounding box.

### Landmark Detection

We saw how to output the bounding box in the previous section, in more general case, we can have the NN output just X, Y coordinates of important point in the image. Those are sometimes called landmarks. Let's see some examples:

For example, if you are working in a face recognition problem you might want some points on the face like corners of the eyes, corners of the mouth, and corners of the nose and so on. This can help in a lot of application like detecting the pose of the face. Another application is when you need to get the skeleton of the person using different landmarks/points in the person which helps in some applications.

The Y shape would be something like this:

![](https://i.ibb.co/m0mGNNW/image.png)

Note that in your labeled data, if l1x,l1y is the left corner of left eye, all other l1x,l1y of the other examples has to be the same.

### Object Detection

We are going to use a technique called **the sliding window** algorithm to solve object detection.

Let's take an example of car detection.
First step would be to train a CNN on cropped car image and non car image so that the model can tell us if one image is a car or not. Then with the sliding windws technique we will :

- decide of a window size.
- split the image into rectangles of the size of the window. The goal is to slide the window on every region of the image so that every part of it is covered. 
- Pick other size of window and do the whole thing again.
- give to the CNN the windows to see if its detect a car.
- store the windows that contains cars
- If two or more rectangles intersects choose the rectangle with the best accuracy.

Disadvantage of sliding window is the computation time.
In the era of machine learning before deep learning, people used a hand crafted linear classifiers that classifies the object and then use the sliding window technique. The linear classier make it a cheap computation. But in the deep learning era that is so computational expensive due to the complexity of the deep learning model.
To solve this problem, we can implement the sliding windows with a Convolutional approach.  One other idea is to compress your deep learning model.


### Convolutional Implementation of Sliding Windows

To solve the computation time problem of the sliding window, we can implement the it with a Convolutional approach.

So the first thing to do is turning FC layers into Conv layers using a convolution :
![](https://i.ibb.co/yR9DmSp/image.png)

The conv implementation is very simple :

![](https://i.ibb.co/vD7vQrS/image.png)
First lets consider that the Conv net you trained is like the first row NN (No FC all is conv layers).

Say now we have a 16 x 16 x 3 image that we need to apply the sliding windows in. By the normal implementation that have been mentioned in the section before this, we would run this Conv net four times each rectangle size will be 16 x 16.

The convolution implementation will be as follows:
- Simply we have feed the image into the same Conv net we have trained.
- The left cell of the result "The blue one" will represent the the first sliding window of the normal implementation.
- The other cells will represent the others.
- Its more efficient because it now shares the computations of the four times needed.

The last example (last row)  has a total of 16 sliding windows that shares the computation together.

The weakness of the algorithm is that the position of the rectangle wont be so accurate. Maybe none of the rectangles is exactly on the object you want to recognize.

### Bounding Box Predictions

We know the weakness of the sliding windows: the bounding box are not very accurate because when we sliding the windows there are chances none of the boxes really match up perfectly with the position of the car. The best match would probably be one that we draw.

How to get our bounding box prediction more accurate?

For that, let use the YOLO algorithm. Its stands for **you only look once** and was developed back in 2015.

Here the steps of the Yolo Algorithm:
-  Lets say we have an image of 100 X 100
-  Place a 3 x 3 grid on the image. For more smother results you should use 19 x 19 for the 100 x 100
- Apply the classification and localization algorithm we discussed in a previous section to each section of the grid. bx and by will represent the center point of the object in each grid and will be relative to the box so the range is between 0 and 1 while bh and bw will represent the height and width of the object which can be greater than 1.0 but still a floating point value.
-  Do everything at once with the convolution sliding window. If Y shape is 1 x 8 as we discussed before then the output of the 100 x 100 image should be 3 x 3 x 8 which corresponds to 9 cell results.
-  Merging the results using predicted localization mid point.

Note that we will have a problem if we have found more than one object in one grid box.

- One of the best advantages that makes the YOLO algorithm popular is that it has a great speed and a Conv net implementation.
How is YOLO different from other Object detectors? YOLO uses a single CNN network for both classification and localizing the object using bounding boxes.

In the next sections we will see some ideas that can make the YOLO algorithm better.

### Intersection Over Union

How do you tell if your object detection algorithm is working well? 

**Intersection Over Union** is a function used to evaluate the object detection algorithm.

![](https://i.ibb.co/CvX3XcK/image.png)
The algorithm computes size of intersection and divide it by the union. More generally, IoU is a measure of the overlap between two bounding boxes.

In the previous example, the red is the labeled output and the purple is the predicted output. Let's compute the IoU, if the IoU is >= to 0.5 (human convention) then we can say that our prediction is correct. 

Of course, if the IoU is equal to 1, then that means that the two bounding box are overleaping. So the higher the IoU, the better is the accuracy.

### Non-max Suppression

One of the problems of Object Detection as you've learned about this so far, is that your algorithm may find multiple detections of the same objects. Rather than detecting an object just once, it might detect it multiple times.

Non-max suppression is a way for you to make sure that your algorithm detects each object only once.

For example, the car detection :

![](https://i.ibb.co/xm3hSLN/image.png)

Because we're running the image classification and localization algorithm on every grid cell ( 19 x 19 = 361 grid cells) it's possible that many of them detect an object. So, when you run your algorithm, you might end up with multiple detections of each object.

What non-max suppression does is it cleans up these detections so they end up with just one detection per car rather than multiple detections per car.

non-max means that you're going to output your maximal probabilities classifications (Pc) but suppress the close ones that are non-maximal. Hence the name, non-max suppression.

Here the algorithm steps:

Lets assume that we are targeting one class as an output class.
- Y shape should be [Pc, bx, by, bh, hw] Where Pc is the probability if that object occurs.
-  Discard all boxes with Pc < 0.6
-  While there are any remaining boxes:
	-  Pick the box with the largest Pc Output that as a prediction.
	-  Discard any remaining box with IoU > 0.5 with that box output in the previous step i.e any box with high overlap(greater than overlap threshold of 0.5).

If there are multiple classes/object types c you want to detect, you should run the Non-max suppression c times, once for every output class.

### Anchor Boxes

One of the problems with object detection as we have seen it so far is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects?

You can use the idea of anchor boxes. Let's start with an example.

Say we have a pedestrian and a car in the same gird (even if it's happens rarely) like in this image:

If Y = [Pc, bx, by, bh, bw, c1, c2, c3] Then to use two anchor boxes like this: Y = [Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3] We simply have repeated the one anchor
Y.

We will chose two anchor box shaped as the shape of our object.

![](https://i.ibb.co/vXtWX4r/image.png)
So Previously, each object in training image is assigned to grid cell that contains that object's midpoint. With two anchor boxes, Each object in training image is assigned to grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU. You have to check where your object should be based on its rectangle closest to which anchor box.

You may have two or more anchor boxes but you should know their shapes. how do you choose the anchor boxes and people used to just choose them by hand. Maybe five or ten anchor box shapes that spans a variety of shapes that cover the types of objects you seem to detect frequently.
You may also use a k-means algorithm on your dataset to specify that.
Anchor boxes allows your algorithm to specialize, means in our case to easily detect wider images or taller ones.

### YOLO Algorithm

Let's put all the pieces together to form the YOLO algorithm - a state-of-the-art object detection model that is fast and accurate.

Suppose we need to do object detection for our autonomous driver system.It needs to identify three classes:

	i. Pedestrian (Walks on ground).
	ii. Car.
	iii. Motorcycle.
	
We decided to choose two anchor boxes, a taller one and a wide one.

Like we said in practice they use five or more anchor boxes hand made or generated using k-means.

Our labeled Y shape will be [Ny, HeightOfGrid, WidthOfGrid, 16] , where Ny is number of instances and each row (of size 16) is as follows: [Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3]

Your dataset could be an image with a multiple labels and a rectangle for each label, we should go to your dataset and make the shape and values of Y like we agreed.

![](https://i.ibb.co/5FbvnFB/image.png)
We first initialize all of them to zeros and ?, then for each label and rectangle choose its closest grid point then the shape to fill it and then the best anchor point based on the IOU. so that the shape of Y for one image should be [HeightOfGrid, WidthOfGrid,16]

Train the labeled images on a Conv net. you should receive an output of [HeightOfGrid, WidthOfGrid,16] for our case.

To make predictions, run the Conv net on an image and run Non-max suppression algorithm for each class you have in our case there are 3 classes.

![](https://i.ibb.co/BgqKTrz/image.png)

When running the Non-max suppression, in the first step we might have a lot of bounding box, by removing the low probability predictions you should have less boxes. Finally we get the best bounding boxes by applying the IoU filter. 

Note that YOLO is not good at detecting smaller object.

You can find implementations for YOLO here:

https://github.com/allanzelener/YAD2K
https://github.com/thtrieu/darkflow
https://pjreddie.com/darknet/yolo/


### Region Proposals (R-CNN)

R-CNN is an algorithm that also makes an object detection. Yolo tells that its faster:

> Our model has several advantages over classifier-based systems. It
> looks at the whole image at test time so its predictions are informed
> by global context in the image. It also makes predictions with a
> single network evaluation unlike systems like R-CNN which require
> thousands for a single image. This makes it extremely fast, more than
> 1000x faster than R-CNN and 100x faster than Fast R-CNN. See our paper
> for more details on the full system.

But one of the downsides of YOLO that it process a lot of areas where no objects are present.
- R-CNN stands for regions with Conv Nets.
- R-CNN tries to pick a few windows and run a Conv net (your confident classifier) on top of them.
- The algorithm R-CNN uses to pick windows is called a segmentation algorithm. Outputs something like this:

![](https://i.ibb.co/fp0zcYR/image.png)

If for example the segmentation algorithm produces 2000 blob then we should run our classifier/CNN on top of these blobs.

There has been a lot of work regarding R-CNN tries to make it faster:
- R-CNN:
Propose regions. Classify proposed regions one at a time. Output label + bounding box.
Downside is that its slow.

- Fast R-CNN:
Propose regions. Use convolution implementation of sliding windows to classify all the proposed regions.

- Faster R-CNN:
Use convolutional network to propose regions.

- Mask R-CNN:
Most of the implementation of faster R-CNN are still slower than YOLO.
Andrew Ng thinks that the idea behind YOLO is better than R-CNN because you are able to do all the things in just one time instead of two times.

Other algorithms that uses one shot to get the output includes SSD and MultiBox.
R-FCN is similar to Faster R-CNN but more efficient.

#### What you should remember:[](https://godrdpxw.labs.coursera.org/notebooks/week3/Car%20detection%20for%20Autonomous%20Driving/Autonomous_driving_application_Car_detection_v3a.ipynb#What-you-should-remember:)

-   YOLO is a state-of-the-art object detection model that is fast and accurate
-   It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
-   The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
-   You filter through all the boxes using non-max suppression. Specifically:
    -   Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
    -   Intersection over Union (IoU) thresholding to eliminate overlapping boxes
-   Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.

## Special applications: Face recognition & Neural style transfer

By now, we've learned a lot about ConvNet. This week, we will see important special applications of ConvNet.
We'll start the face recognition, and then go on later this week to Neural Style transfer, which you get to implement in the problem exercise as well to create your own artwork

### Face Recognition

#### What is Face recognition

Face recognition system identifies a person's face. It can work on both images or videos.
What Andrew show us in his example is a face recognition + liveness detection : its prevents the network from identifying a face in an image when it's not a real person. It can be learned by supervised deep learning.

Face verification vs. face recognition:
- Verification:
	- Input: image, name/ID. (1 : 1)
	- Output: whether the input image is that of the claimed person.
	- "is this the claimed person?"
- Recognition (more complex):
	- Has a database of K persons
	- Get an input image
	- Output ID if the image is any of the K persons (or not recognized)
	- "who is this person?"

We can use a face verification system to make a face recognition system. The accuracy of the verification system has to be high (around 99.9% or more) to be use accurately within a recognition system because the recognition system accuracy will be less than the verification system given K persons.

#### One shot Learning

One of the challenges of face recognition is that you need to solve the one-shot learning problem. What that means is that for most face recognition applications you need to be able to recognize a person given just one single image, or given just one example of that person's face. And, historically, deep learning algorithms don't work well if you have only one training example. Let's see an example of what this means and talk about how to address this problem.

One Shot Learning: A recognition system is able to recognize a person, learning from one image.

Say you have one image of every of your employee/colleague, your system have to check is a person is one of your employee/colleague or none of them.  What we can do is training a ConvNet with a softmax output of (#employee/colleague + 1)  (the +1 is for none of the classes) layer, but knowing that we only have one image of each, we know that this not going to work correctly.

So what we are going to do instead is learning a **similarity function** and in particular :

    d( img1, img2 ) = degree of difference between images.

We want d result to be low in case of the same faces and high if the two face are different.

We will use (tau) T as a threshold for d (so T will be a hyperparameters):

    If d( img1, img2 ) <= T Then the faces are the same.

The plus of this method is that if we have a new person in our team, then we don't have to re-train the whole model. Instead, we just need to add that person image to our data set, and the whole system will work find. So the solution to One shot learning problem is solved by the similarity function.


#### Siamese Network

We will implement the similarity function using a type of NNs called Siamease Network in which we can pass multiple inputs to the two or more networks with the same architecture and parameters.
Siamese network architecture are as the following:

![](https://i.ibb.co/SxdS5ZR/image.png)


The loss function will be `d(x1, x2) = || f(x1) - f(x2) ||^2`
If X1 , X2 are the same person, we want d to be low. If they are different persons, we want d to be high.

#### Triplet Loss

One way to learn the parameters of the neural network so that it gives you a good encoding for your pictures of faces is to define an applied gradient descent on the triplet loss function. Let's see what that means.

Triplet Loss is one of the loss functions we can use to solve the similarity distance in a Siamese network. Our learning objective in the triplet loss function is to get the distance between an Anchor image and a positive or a negative image. (Positive means same person, while negative means different person.)

Note that the triplet name came from that we are comparing an anchor A with a positive P and a negative N image.

![](https://i.ibb.co/8Xkh810/image.png)

Formally we want that the positive distance to be less than negative distance. To make sure the NN won't get an output of zeros easily we add an alpha margin (a small function).

The final Loss function given 3 images (A, P, N), would be :

![](https://i.ibb.co/zh3qXcw/image.png)

You need multiple images of the same person in your dataset. Then get some triplets out of your dataset. Dataset should be big enough.

Choosing the triplets A, P, N:
- During training if A, P, N are chosen randomly (Subjet to A and P are the same and A and N aren't the same) the one of the problems this constrain is easily satisfied `d(A, P) + alpha <= d (A, N)` So the NN wont learn much.

- What we want to do is choose triplets that are hard to train on. So for all the triplets we want this to be satisfied:

    d(A, P) + alpha <= d (A, N)

Note that commercial recognition systems are trained on a large datasets like 10/100 million images. There are a lot of pretrained models and parameters online for face recognition. The best to do if we have an application of face recognition to build is to get the weight of these systems.

### Face Verification and Binary Classification

The Triplet Loss is one good way to learn the parameters of a continent for face recognition. There's another way to learn these parameters. Let see how face recognition can also be posed as a straight binary classification problem.

So to learn the parameters another way, we can :

![](https://i.ibb.co/QkqxgQs/image.png)

The final layer can be a sigmoid function for example :

    Y' = wi * Sigmoid ( f(x(i)) - f(x(j)) ) + b

 where the subtraction is the Manhattan distance between f(x(i)) and f(x(j))
Some other similarities can be Euclidean and Ki square similarity.

The NN here is Siamese means the top and bottom convs has the same parameters.

A good performance/deployment trick:
- Pre-compute all the images that you are using as a comparison to the vector f(x(j))
- When a new image that needs to be compared, get its vector f(x(i)) then put it with all the pre computed vectors and pass it to the sigmoid function.
- This version works quite as well as the triplet loss function.


Available implementations for face recognition using deep learning includes:
- Openface
- FaceNet
- DeepFace

### Neural style transfer

#### What is neural style transfer?

This is one of the more exciting application of ConvNet.

Neural style transfer takes a content image C and a style image S and generates the content image G with the style of style image.

![](https://i.ibb.co/hFmS9J1/image.png)
In order to implement Neural Style Transfer, we need to look at the features extracted by ConvNet at various layers : the shallow and the deeper layers of a ConvNet.

Before diving into how we can implement a Neural Style Transfer, we will try to have a better intuition about whether all these layers of a ConvNet really computing.

#### What are deep ConvNets learning?

What are deep ConvNets really learning?

We will see some visualizations that will help us with our intuition about what the deeper layers of a ConvNet really are doing.

And this will help us think through how you to implement neural style transfer as well.


As this part was a little bit confusing for me, let first clarify some things.

A hidden unit in a Conv layer is (from what I understand) is a pixel of the matrix  (channel stacked) resulting from a conv operation + b that we applied the ReLu activation function. To write it more clearly :

**In a ConvNet, a given unit in a hidden layer is the output of the application of the filter at a particular point in the input space.**

From [here](https://stats.stackexchange.com/questions/333099/definition-of-hidden-unit-in-a-convnet) , we can note that A hidden unit, in general, has an operation **Activation(W*X+b)**. Note that when implementing the CONV layer in keras or  tensorflow we use :
```
tf.nn.Conv2D(**hidden_units**,...)
```
Here it's the filter shape we use.

To summarize this part: 

The number of filters in one layer is the number of neurons in that layer. The number of filters is also the number of channels for the output of that layer, and one pixel in the output (in one channel) is the output of a neuron.

Also note that from a mentor in Coursera classes :  A patch is a part of the image that is covered by the filter.

Knowing all that, let's continue with an example:

We want to visualize what the hidden units in different layers are computing. That what we can do :

![](https://i.ibb.co/Cm2bFq9/image.png)

- Pick a unit in layer l. Find the nine image patches that maximize the unit's activation. So in other words pass your training set through your neural network, and figure out what is the image that maximizes that particular unit's activation.
Notice that a hidden unit in layer one will see relatively small portion of NN, so if you plotted it it will match a small image in the shallower layers while it will get larger image in deeper layers.
- Repeat for other units and layers.
- It turns out that layer 1 are learning the low level representations like colors and edges.

Note : what we did, is just taking the result of the first conv layers, of one channel (so the result of one filter) we looked at n patches (the part of the image that was convolved with the filter) where the activation function was the max.

In the deeper layers, a hidden unit will see a larger region of the image. Where at the extreme end each pixel could hypothetically affect the output of these later layers of the neural network. So later units are actually seen larger image patches.


![](https://i.ibb.co/nCxL2tZ/image.png)
- The first layer was created using the weights of the first layer. Other images are generated using the receptive field in the image that triggered the neuron to be max.

Mahmoud Badry notes suggests this [link](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807) 

#### Cost Function

To build a Neural Style Transfer system, let's define a cost function for the generated image.

Remember the problem : Give a content image C, a style image S, and a generated image G.

What we're going to do is define a cost function J(G) that measures how good is a particular generated image and we'll use gradient to descent to minimize J(G) in order to generate this image :

    J(G) = alpha * J(C,G) + beta * J(S,G)

- J(C, G) measures how similar is the generated image to the Content image.
- J(S, G) measures how similar is the generated image to the Style image.
- alpha and beta are relative weighting to the similarity and these are hyperparameters. We use the same convention as the author of the paper.

The way the algorithm would run is as follows : having to find the cost function J(G) in order to actually generate a new image what you do is the following:

- initialize the generated image G randomly. (For example G: 100 X 100 X 3)
- Use gradient descent to minimize J(G)
`G = G - dG` We compute the gradient image and use gradient decent to minimize the cost function. This will update the values of the images G.

Here an example :

![](https://i.ibb.co/rsXjLFh/image.png)

#### Content Cost Function

The cost function of the neural style transfer algorithm had a content cost component and a style cost component. Let's start by defining the content cost component.

![](https://i.ibb.co/MM9j1c7/image.png)

If we choose l to be small (like layer 1), we will force the network to get similar output to the original content
image. In practice l is not too shallow and not too deep but in the middle.

So, what we'll do is define J_content(C,G) as just how different are these two activations. So, we'll take the element-wise difference between these hidden unit activations in layer l, between when you pass in the content image compared to when you pass in the generated image, and take that squared.

#### Style Cost Function

Meaning of the style of an image:
- Say you are using layer l's activation to measure style.
Define style as correlation between activations across channels.
That means given an activation like this:

![](https://i.ibb.co/fQNTh5Y/image.png)

How correlate is the orange channel with the yellow channel?

- Correlated means if a value appeared in a specific channel a specific value will appear too (Depends on each
other).
- Uncorrelated means if a value appeared in a specific channel doesn't mean that another value will appear (Not
depend on each other)

- The correlation tells you how a components might occur or not occur together in the same image.
- The correlation of style image channels should appear in the generated image channels.

Style matrix (Gram matrix):

- Let a(l)[i, j, k] be the activation at l with (i=H, j=W, k=C) Also G(l)(s) is matrix of shape nc(l) x nc(l)
- We call this matrix style matrix or Gram matrix.
- In this matrix each cell will tell us how correlated is a channel to another channel.
-To populate the matrix we use these equations to compute style matrix of the style image and the generated image.

![](https://i.ibb.co/Krrqv8D/image.png)
- As it appears its the sum of the multiplication of each member in the matrix.
To compute gram matrix efficiently:
- Reshape activation from H X W X C to HW X C
 - Name the reshaped activation F.

    G[l] = F * F.T

Finally the cost function will be as following:

    J(S, G) at layer l = (1/ 2 * H * W * C) || G(l)(s) - G(l)(G) ||

And if you have used it from some layers

    J(S, G) = Sum (lamda[l]*J(S, G)[l], for all layers)

To summarize

Steps to be made if you want to create a tensorflow model for neural style transfer:
i. Create an Interactive Session.
ii. Load the content image.
iii. Load the style image
iv. Randomly initialize the image to be generated
v. Load the VGG16 model
vi. Build the TensorFlow graph:
	- Run the content image through the VGG16 model and compute the content cost
	- Run the style image through the VGG16 model and compute the style cost
	- Compute the total cost
	- Define the optimizer and the learning rate
vii. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.

#### 1D and 3D Generalizations

So far we have used the Conv nets for images which are 2D.

Conv nets can work with 1D and 3D data as well.

An example of 1D convolution:
- Input shape (14, 1)
- Applying 16 filters with F = 5 , S = 1
- Output shape will be 10 X 16
- Applying 32 filters with F = 5, S = 1
- Output shape will be 6 X 32

The general equation (N - F)/S + 1 can be applied here but here it gives a vector rather than a 2D matrix.

1D data comes from a lot of resources such as waves, sounds, heartbeat signals.

In most of the applications that uses 1D data we use Recurrent Neural Network RNN.

3D data also are available in some applications like CT scan.
Example of 3D convolution:
- Input shape (14, 14,14, 1)
- Applying 16 filters with F = 5 , S = 1
- Output shape (10, 10, 10, 16)
- Applying 32 filters with F = 5, S = 1
- Output shape will be (6, 6, 6, 32)


Notebook notes

Face recognition problems commonly fall into two categories:

-   **Face Verification** - "is this the claimed person?". For example, at some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person. A mobile phone that unlocks using your face is also using face verification. This is a 1:1 matching problem.
-   **Face Recognition** - "who is this person?". For example, the video lecture showed a [face recognition video](https://www.youtube.com/watch?v=wr4rx0Spihs) of Baidu employees entering the office without needing to otherwise identify themselves. This is a 1:K matching problem.

FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. By comparing two such vectors, you can then determine if two pictures are of the same person.

