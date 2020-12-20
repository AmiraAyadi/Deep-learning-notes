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


