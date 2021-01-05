## Recurrent Neural Networks

### Why sequence models

Models like recurrent neural networks or RNNs have transformed speech recognition, natural language processing and other areas. In this course, we will learn how to build these models.
Let's start by looking at a few examples of where sequence models can be useful.

![](https://i.ibb.co/vhp9pTd/image.png)

- Speech recognition (sequence to sequence)
- Music generation (one to sequence)
- Sentiment classification (sequence to one)
- DNA sequence analysis (sequence to sequence)
- Machine translation (sequence to sequence)
- Video activity recognition (sequence to one)
- Name entity recognition (sequence to sequence)

As you can tell from this list of examples, there are a lot of different types of sequence problems.

### Notation

Let's start by defining a notation that we'll use to build up these sequence models.
As a motivation example, let say we want a sequence model to automatically tell you where are the people name in this sentence :

X : "Harry Potter and Hermione Granger invented a new spell"

and we want the model to output :

Y: 1 1 0 1 1 0 0 0 0

Both X and Y are (1, 9)

Note that this is a this is a problem called **Named-entity recognition** and this is used by search engines for example, to index all of say the last 24 hours news of all the people mentioned in the news articles. Name into the recognition systems can be used to find people's names, companies names, times, locations, countries names, currency names, and so on in different types of text.

For the notation we are going to use :

- x<sup><i></sub> is the i-th element of x. x<sup><i></sub> for our example is "Harry".
- y<sup><i></sub> is the i-th element of y. y<sup><i></sub> for our y example is 1.
- T<sub>x</sub> is the size of the input sequence and T<sub>y</sub> is the size of the output sequence. T<sub>x</sub> = T<sub>y</sub> = 9 in the last example.
- x<sup>(i)\<t></sup> is the element t of the sequence of input vector i. Similarly y<sup>(i)\<t></sup> means the t-th element in the output sequence of the i training example.
- T<sub>x</sub><sup>(i)</sup> the input sequence length for training example i. It can be different across the examples. Similarly for T<sub>y</sub><sup>(i)</sup> will be the length of the output sequence in the i-th training example.

How to represent individual words in the sequence ? This is one of the challenges of NLP which stands for natural language processing.

First, we need a vocabulary list that contains all the words in our target sets. e sort that by alphabetical order, then for each word we will have a unique index that represent it. 

![](https://i.ibb.co/XLBvQ0j/image.png)

Then we create a one-hot-encoding sequence for each word in the dataset given the vocabulary we used. 

What if you encounter a word that is not in your vocabulary? Well the answer is, you create a new token or a new fake word called Unknown Word which under note UNK to represent words not in your vocabulary.

The goal given this representation is for x to learn a mapping using a sequence model to then target output y as a supervised learning problem.

Note that vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies use even a million.
To build vocabulary list, you can read all the texts you have and get m words with the most occurrence, or search online for m most words that occurs.

### Recurrent Neural Network Model

Now, let's talk about how you can build a model, a neural network to learn the mapping from x to y.

First we could wander why don't just resolve the problem (above) just by a NN? 
There are two problems :
- Inputs, outputs can be different lengths in different examples.
Actually you can think solving this with normal NNs by padding to the maximum lengths but it's not a good representation.
- Doesn't share features learned across different positions of text/sequence.
Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. That's what we will do in RNNs.

A recurrent neural network which we'll start to describe in the next slide does not have either of these disadvantages. So what is it ?  Let's build one.

![](https://i.ibb.co/xfsqRkc/image.png)


Notes :
- x<sup><0></sub> is usually initialize with 0, some research initialize it randomly. There are some other methods to initializing it but doing it with 0 is fine.
- Here T<sub>x</sub> = T<sub>y</sub> but for other problems this can be different.
- In the right part of the picture is an other way to write (represent) the RNN. A lot of papers and books do it like this but it's harder to interpret it and it is easier to roll this drawings to the unrolled version of the left.

The recurrent neural network scans through the data from left to right. The parameters it uses for each time step are shared. the parameters governing the connection from X1 to the hidden layer, will be some set of parameters we're going to write as  W<sub>ax</sub> and is the same parameters W<sub>ax</sub> that it uses for every time step.
The activation (the horizontal connections will be governed by some set of parameters W<sub>aa</sub> , again, the same W<sub>aa</sub> is used on every time step.
Similarly there some W<sub>ya</sub>that governs the output predictions.


![](https://i.ibb.co/3ptQ0n1/image.png)
Here we can see that the weight matrix W is the memory the RNN is trying to maintain from the previous layers.

In the discussed RNN architecture, the current output ŷ depends on the previous inputs and activations. Let's have this example 'He Said, "Teddy Roosevelt was a great president"'. In this example Teddy is a person name but we know that from the word president that came after Teddy not from He and said that were before it. So limitation of the discussed architecture is that it can not learn from elements later in the sequence. To address this problem we will later discuss Bidirectional RNN (BRNN).

Note about the weights:

- We "named" Waa because we multiply it with a to compute a.
- Same for Wax, we multiply it with X to compute a.
- Same again for Wya.
- the length will be :
	-  W<sub>ax</sub>: (NoOfHiddenNeurons, n<sub>x</sub>)
        - W<sub>aa</sub>: (NoOfHiddenNeurons, NoOfHiddenNeurons)
        - W<sub>ya</sub>: (n<sub>y</sub>, NoOfHiddenNeurons)


What the Forward prop look like with RNN?

![](https://i.ibb.co/m9vt5yP/image.png)

The activation function of a is usually tanh or ReLU and for y depends on your task choosing some activation functions like sigmoid and softmax. In name entity recognition task we will use sigmoid because we only have two classes.

Let's simplify all the RNN notation 

![](https://i.ibb.co/611mqjH/image.png)
- w<sub>a</sub> is w<sub>aa</sub> and w<sub>ax</sub> stacked horizontaly.
  - [a<sup>\<t-1></sup>, x<sup>\<t></sup>] is a<sup>\<t-1></sup> and x<sup>\<t></sup> stacked verticaly.
  - w<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
  - [a<sup>\<t-1></sup>, x<sup>\<t></sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)

### Backpropagation through time

We already implemented the basic structure of an RNN and we saw how forward prop works. Let's see how back propagation this time works. Of course, when implementing it with a framework, often the framework will automatically take care of it, but it  is important to know how it's work so let's go !

Here the computation graph :

![](https://i.ibb.co/Q6tvFdC/image.png)

-  w<sub>a</sub>, b<sub>a</sub>, w<sub>y</sub>, and b<sub>y</sub> are shared across each element.
- Here we used cross-entropy loss function for the example.
- We will sum over all the calculated single example losses to get the loss for one example.
- The back propagation here is called back propagation through time because we pass activation a from one sequence element to another like backwards in time.

### Different types of RNNs

So far we have seen only one RNN architecture in which T<sub>x</sub> equals T<sub>Y</sub>. In some other problems, they may not equal so we need different architectures. It turns out that we could modify the basic RNN architecture to address all of these problems.

The presentation in this video was inspired by a blog post by Andrej Karpathy [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) titled, The Unreasonable Effectiveness of Recurrent Neural Networks.

Let's go through some examples:

![](https://i.ibb.co/MMdLYTG/image.png)

Note that the architecture we already saw is called many-to-many.

- An example of a **many-to-one** architecture is sentiment analysis problem (like if X is a sentence and Y a rate, an integer from 0 to 5).
- An example of a **one-to-many** would be music generation. Note that starting the second layer we are feeding the generated output back to the network.
- The **one-to-one** architecture is basically an NN.
- There are two **many-to-many** architecture: the one we already saw and one more interesting which is when the input and the output length are different.
- Applications like machine translation inputs and outputs sequences have different lengths in most of the cases. That what it would look like :

![](https://i.ibb.co/kxL8VpD/image.png) 

And so, this that collinear network architecture has two distinct parts. There's the encoder which takes as input, say a French sentence, and then, there's is a decoder, which having read in the sentence, outputs the translation into a different language.

To summarize : 


![](https://i.ibb.co/Fs0vZ0D/image.png)

We will master all these different RNN by the end of the courses. 

### Language model and sequence generation

Language modeling is one of the most basic and important tasks in natural language processing. There's also one that RNNs do very well. In this course, we will learn about how to build a language model using an RNN.

What is a language model ? 

Let's say we are solving a speech recognition problem and someone says a sentence that can be interpreted into to two sentences:

- The apple and **pair** salad
- The apple and **pear** salad

Pair and pear sounds exactly the same, so how would a speech recognition application choose from the two. That's where the language model comes in. It gives a probability for the two sentences and the application decides the best based on this probability.

How build a language model with RNN?

- You would first need a training set comprising a large corpus of english text. Or text from whatever language you want to build a language model of. The word **corpus** is an NLP terminology that just means a large body or a very large set of english text of english sentences.
- The second thing you would do is tokenize this sentence, that means you would form a vocabulary and then map each of these words to one-hot vectors.
- One thing you might also want to do is model when sentences end. To do this, you can add an extra token called a <EOS>. That stands for End Of Sentence that can help you figure out when a sentence ends.
- If there are word not in your vocabulary, then use an <UNK>. 

Note that when doing the tokenization step, you can decide whether or not the period should be a token as well. In this example down here, we are just ignoring punctuation.

Given the sentence "Cats average 15 hours of sleep a day. <EOS> ", let's build the RNN model:

![](https://i.ibb.co/VQgfz4r/image.png)

- a<sup><0></sup> will be initialize at 0.
- In the first step, a<sup><1></sup> will make a softmax prediction to try to figure out what is the probability of the first word y<sup><1></sup>. What is the probability of any word in the dictionary? what the chance of the first word is "Aron" ? etc. This would be a 10.000 (number of our vocabulary) word softmax output.
- Then, the RNN steps forward to the next step and has some activation, a<sup><1></sup> to the next step. And at this step, his job is try figure out, what is the second word? But now we will also give it the correct first word etc...
- The loss function is defined by cross-entropy loss. `i` is for all elements in the corpus, `t` - for all time steps.

To use this model:
i. For predicting the chance of next word, we feed the sentence to the RNN and then get the final y hot vector and sort it by maximum probability.
ii. For taking the probability of a sentence, we compute this:

    p(y , y , y ) = p(y ) * p(y | y ) * p(y | y , y )

This is simply feeding the sentence into the RNN and multiplying the probabilities (outputs).

### Sampling novel sequences

After you train a sequence model, one of the ways you can informally get a sense of what is learned is to have a sample novel sequences.
Let's take a look at how you could do that.

![](https://i.ibb.co/D527p0Y/image.png)

So the network was trained with the top structure. To sample, we must do something a little different : 
- The first goal is to sample our first word we want the model to generate. We first pass a<sup><0></sup> = zeros vector, and x<sup><1></sup> = zeros vector.
- After the first step, we should have some softmax probabilities over possible outputs. we take this vector and use, for example, the numpy command `np.random.choice` to sample according to distribution defined by this vector probabilities, and that lets us sample the first words.
- We pass the last predicted word with the calculated a<sup><1></sup>
- We keep doing 3 & 4 steps for a fixed length or until we get the <EOS> token. Or alternatively, if you do not include this in your vocabulary then you can also just decide to sample 20 words or 100 words or something, and then keep going until you've reached that number of time steps.
- You can reject any <UNK> token if you mind finding it in your output.


So far we've been building a words level RNN, It's also possible to implement a character-level language model. In the character-level language model, the vocabulary will contain [azA-Z0-9] , punctuation, special characters and possibly token. You can also see the characters in your vocabulary set and add them here.

Note that if you build a character level language model rather than a word level language model, then your sequence y1, y2, y3, would be the individual characters in your training data, rather than the individual words in your training data.

Character-level language model has some pros and cons compared to the word-level language model. The advantage is it will be no <UNK> token - it can create any word.
The main disadvantage taught is that you end up with much longer sequences.The Character-level language models are not as good as word-level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence. They are also more computationally expensive and harder to train.

The trend Andrew has seen in NLP is that for the most part, a word-level language model is still used, but as computers get faster there are more and more applications where people are, at least in some special cases, starting to look at more character-level models. Also, they are used in specialized applications where you might need to deal with unknown words or other vocabulary words a lot. Or they are also used in more specialized applications where you have a more specialized vocabulary.

### Vanishing gradients with RNNs


We've learned about how RNNs work and how they can be applied to problems like name entity recognition, as well as to language modeling, and we saw how back propagation can be used to train in RNN.
It turns out that one of the problem with the basic RNN, is that it's runs into a vanishing gradient problems. Let's discuss that.

Suppose we are working with language modeling problem and there are two sequences that model tries to learn:

- "The cat, which already ate ..., was full"
- "The cats, which already ate ..., were full"

Note that the dots represent many words in between.
What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN is not very good at capturing very long-term dependencies like this.

To explained why, remember the what we said on vanish gradient for very deep NN : you would carry out forward prop, from left to right and then back prop. As it is a very deep neural network, then the gradient from just output y, would have a very hard time propagating back to affect the weights of the earlier layers. It is the same for RNN. So an RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to
optimize and **therefore some of the weights may not be updated properly.**

What this means is, it might be difficult to get a neural network to realize that it needs to memorize the singular noun or a plural noun, so that later on in the sequence that can generate either was or were, depending on whether it was singular or plural. So the RNNs aren't good in long-term dependencies.

We're doing back prop, the gradients could not just decrease exponentially, it may also increase exponentially with the number of layers you go through. So exploding gradients is also a problem. 

It turns out that vanishing gradients tends to be the bigger problem with training RNNs, although when its happens, it can be catastrophic because the exponentially large gradients can cause your parameters to become so large that your neural network parameters get really messed up  and you might often see NaNs results of a numerical overflow in your neural network computation.

If you do see exploding gradients, one solution to that is apply gradient clipping. that means is look at your gradient vectors, and if it is bigger than some threshold, re-scale some of your gradient vector so that is not too big. So there are clips according to some maximum value.
So if you see exploding gradients, if your derivatives do explode or you see NaNs, just apply gradient clipping, and that's a relatively robust solution that will take care of exploding gradients.

Vanishing gradients is much harder to solve and it will be the subject of the next few courses.

Extra from [here](https://github.com/ashishpatel26/Andrew-NG-Notes/edit/master/andrewng-p-5-sequence-models.md):
- Solutions for the Exploding gradient problem:
	- Truncated backpropagation.
		- Not to update all the weights in the way back.
		- Not optimal. You won't update all the weights.
	- Gradient clipping.
- Solution for the Vanishing gradient problem:
	- Weight initialization.
		- Like He initialization.
	- Echo state networks.
	- Use LSTM/GRU networks.
		- Most popular.
		- We will discuss it next.

### Gated Recurrent Unit (GRU)

Gated Recurrent Unit which is a modification to the RNN hidden layer that makes it much better capturing long range connections and helps a lot with the vanishing gradient problems. Let's take a look.

An picture representation of a single RNN unit of a hidden layer looks like this :

![](https://i.ibb.co/pJt94g1/image.png)

We are going to use a similar picture representation to explain the GRU.

Given a sentence in singluar like the one above "The cat, which already ate ........................, was full"

Each layer in **GRUs**  has a new variable `C` which is the memory cell. It can tell to whether memorize something or not.  The `C` goal is to have some memory and thus to remember if we should use "was" or "were".


So here the equations :

![](https://i.ibb.co/pjxK6vD/image.png)


In GRU, C<sup>\<t></sup> = a<sup>\<t></sup>.

At every time-step, we're going to consider overwriting the memory cell with a value C<sup>~\<t></sup>. This is going to be a candidate for replacing C<sup><t></sup>.

C<sup>~\<t></sup> is equal to a thanh activation function applied to the weights times the previous activation function and input value X<sup><t></sup> plus the bias.


The key - a really the important idea of the GRU will be that we have a gate called Gamma(u).

As Gamma(u) is a result of a sigmid function, then it will always have a value between 0 and 1 but to have a better intuition, consider it ehter 1 or 0.

So we have come up with a candidate where we're thinking of updating C<sup><t></sup> using C<sup>~\<t></sup>, and then the gate will decide whether or not we actually update it.

The actual value of C<sup><t></sup> will be the result of the equation above. So we update the memory cell based on the update cell and the previous cell.

Here the picture representation of GRU

![](https://i.ibb.co/thQHh6S/image.png)
 Lets take the cat sentence example and apply it to understand this equations:
 
  - We will suppose that U is 0 or 1 and is a bit that tells us if a singular word needs to be memorized.

  - Splitting the words and get values of C and U at each place:

    - | Word    | Update gate(U)             | Cell memory ( C ) |
      | ------- | -------------------------- | --------------- |
      | The     | 0                          | val             |
      | cat     | 1                          | new_val         |
      | which   | 0                          | new_val         |
      | already | 0                          | new_val         |
      | ...     | 0                          | new_val         |
      | was     | 1 (I don't need it anymore)| newer_val       |
      | full    | ..                         | ..              |


Because the update gate U is usually a small number like 0.00001, GRUs doesn't suffer the vanishing gradient problem.

- Shapes:
  - a<sup>\<t></sup> shape is (NoOfHiddenNeurons, 1)
  - c<sup>\<t></sup> is the same as a<sup>\<t></sup>
  - c<sup>~\<t></sup> is the same as a<sup>\<t></sup>
  - u<sup>\<t></sup> is also the same dimensions of a<sup>\<t></sup>
- The multiplication in the equations are element wise multiplication.

We saw the simple GRU, let's see the full example:

![](https://i.ibb.co/bNmDqgS/image.png)

We added Gamma(r), the relevance gate.
So this gate gamma(r) tells you how relevant is c<sup>\<t-1></sup> to computing the next candidate for c<sup>\<t></sup>.

So as you can imagine there are multiple ways to design these types of neural networks. And why do we have gamma r? Why not use a simpler version from the previous slides? So it turns out that over many years researchers have experimented with many, many different possible versions and the GRU is one of the most commun. The other one is the LSTM. Let's talk about them.

Note : in the last equation there is one mistake, it's "time" and not "plus" in the last term.


### Long Short Term Memory (LSTM)

We learned about the GRU, the gated recurrent units, and how that can allow us to learn very long range connections in a sequence. The other type of unit that allows us to do this very well is the LSTM or the long short term memory units, and this is even more powerful than the GRU. Let's take a look.



note that in LSTM, we no longuer have C<sup>\<t></sup> != a<sup>\<t></sup>.

![](https://i.ibb.co/z7Kdhdy/image.png)

In GRU we have an update gate `U`, a relevance gate `r`, and a candidate cell variables C<sup>\~\<t></sup> while in LSTM we have an update gate `U` (sometimes it's called input gate I), a forget gate `F`, an output gate `O`, and a candidate cell variables C<sup>\~\<t></sup>.

Here the picture representation of LSTM

![](https://i.ibb.co/fN2B6np/image.png)

- Some variants on LSTM includes:
  - LSTM with **peephole connections**.
    - The normal LSTM with C<sup>\<t-1></sup> included with every gate.
- There isn't a universal superior between LSTM and it's variants. One of the advantages of GRU is that it's simpler and can be used to build much bigger network but the LSTM is more powerful and general.

Andrew also mentioned this [link](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### Bidirectional RNN

We've seen most of the "cheap" building blocks of RNNs. But, there are just two more ideas that let you build much more powerful models. One is bidirectional RNNs, which lets you at a point in time to take information from both earlier and later in the sequence, so we'll talk about that in this part. The other is deep RNNs and it's the topic of the last part of this week course.

We already see that exampl of the Name entity recognition task :

![](https://i.ibb.co/cgqFSwD/image.png)

So this is a unidirectional or forward directional only RNN.
And, this (and the problem) is true whether these cells are standard RNN blocks or whether they're GRU units or whether they're LSTM blocks.

So what a bidirectional RNN does or BRNN is fix this issue.

Here the architecture of the BRNN (Bidirectional RNN) :

![](https://i.ibb.co/Ry7jCNb/image.png)

What we do is adding a backward reccurents layer to our classic recurrent layers.

Note that this network defines a Acyclic graph.

So given an input sequence, X<sub>1</sub> through  X<sub>4</sub>, the fourth sequence will first compute A forward one, then use that to compute A forward two,
then A forward three, then A forward four. Whereas, the backward sequence would start by computing A backward four, and then go back and compute A backward three and so on. So  the part of the forward propagation goes from left to right, and part - from right to left. It learns from both sides.

To make the prediction, we will compute y&#770;<sup>\<t></sup> using both A forward t and A backward t :

![](https://i.ibb.co/m07XzK6/image.png)
The disadvantage of BiRNNs that you need the entire sequence before you can process it. For example, in live speech recognition if you use BiRNNs you will need to wait for the person who speaks to stop to take the entire sequence and then make your predictions.

For a lot of NLP or text processing problems, a BiRNN with LSTM appears to be commonly used.

### Deep RNNs

The different versions of RNNs you've seen so far will already work quite well by themselves. But for learning very complex functions sometimes is useful to stack multiple layers of RNNs together to build even deeper versions of these models.

Here an example of a deep RNN with 3 layers :

![](https://i.ibb.co/7yhxDcV/image.png)
For RNNs, having three layers is already quite a lot. Because of the temporal dimension, these networks can already get quite big even if you have just a small handful of layers. And you don't usually see these stacked up to be like 100 layers.

One thing you do see sometimes is three recurrent units that connected in time, followed by a network after that. There's a deep network, but that does not have the horizontal connections.

The block in the RNN don't have to be basic RNN, they can be GRU or LSTM.
