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
