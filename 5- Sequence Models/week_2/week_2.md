## Introduction to Word Embeddings

### Word Representation

We will see how the RNN, LSTM and GRU can be applied to NLP - Natural Language Processing, which is one of the features of AI because it's really being revolutionized by deep learning.

One of the key ideas we learn about is word embeddings, which is a way of representing words that lets the algorithm understand analogies between "man" and "woman".

So how do we represent words?

So far we have defined our language by a vocabulary. Then represented our words with a one-hot vector that represents the word in the vocabulary. The weakness of this representation is that it treats a word as a thing that itself and it doesn't allow an algorithm to generalize across words.

![](https://i.ibb.co/Pt4fbxk/image.png)
For example with the sentence "I want a glass of orange ______" A model should predict the next word as juice.
A similar example "I want a glass of apple ______", a model won't easily predict juice here if it wasn't trained on
it. And if so the two examples aren't related although orange and apple are similar.

This is because the inner product between any one-hot encoding vector is zero and the distances between them are the same. So how are we going to explaind to the model that they are closely the "same" ?

Won't it be nice if instead of a one-hot presentation we can instead learn a featurized representation with each of these words, a man, woman, king, queen, apple, orange or really for every word in the dictionary, we could learn a set of features and values for each of them.

![](https://i.ibb.co/0tD3jzb/image.png)
Each word will have a, for example, 300 features with a type of float point number.
  - Each word column will be a 300-dimensional vector which will be the representation.
  - Now, if we return to the examples we described again:
    -  "I want a glass of **orange** ______" 
    -  I want a glass of **apple** ______
  - Orange and apple now share a lot of similar features which makes it easier for an algorithm to generalize between them.
  - We call this representation **Word embeddings**.
over the next few videos we'll find a way to learn words embeddings.

To visualize word embeddings we use a t-SNE (due to Laurens van der Maaten and Geoff Hinton.) algorithm to reduce the features to 2 dimensions which makes it easy to visualize:

![](https://i.ibb.co/tBP8Nc4/image.png)

Notation :

- **O** <sub>idx</sub> is used for any word that is represented with one-hot like in the image.
-  **e**<sub>idx</sub> is used to describe **idx** word features vector.

The word embeddings came from that we need to embed a unique vector inside a n-dimensional space.

### Using word embeddings

We saw what it might mean to learn a featurized representations of different words. In this part of the course, we will see how we can take these representations and plug them into NLP applications. For that we will use one example of name entity recognition problem :

![](https://i.ibb.co/hgJFfXJ/image.png)
"Sally Johnson" is a person's name, after traning the algorithm that uses word embeddings as the inputs, for the second sentence on the example, knowing that orange and apple are very similar (near representation) will make it easier for the learning algorithm to generalize to figure out that 'Robert Lin" is also a person's name.

An interesting case will be what if in our test set we see not "Robert Lin is an apple farmer" but much less common word like "Robert Lin is a durian cultivator"? The answer is yes if you have learned a word embedding that tells you that durian is a fruit and cultivator is like a farmer. So even if the algorithm never seen the word before in the train set.

One of the reasons that word embeddings will be able to do this is **the algorithms to learning word embeddings** can examine very large text found off the Internet - So very large training sets of just unlabeled text. And by examining tons of unlabeled text, which you can download more or less for free, you can figure out that orange and durian are similar and farmer and cultivator are similar, and therefore, learn embeddings, that groups them together

One we have our algorithm learned the word embeddings,  what you can do is then take this word embedding and apply it to your named entity recognition task, for which you might have a much smaller training set. This allow us to do transfer learning. 

To summarize, this is how you carry out transfer learning and word embeddings:
  1. Learn word embeddings from large text corpus (1-100 billion of words).
     (Or download pre-trained embedding online.)
  2. Transfer embedding to new task with the smaller training set (say, 100k words).
  3. Optional: continue to finetune the word embeddings with new data.
     - You bother doing this if your smaller training set (from step 2) is big enough.
- Word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller training set.
- Also, one of the advantages of using word embeddings is that it reduces the size of the input!
  - 10,000 one hot compared to 300 features vector.

Finally, word embeddings has an interesting relationship to the face encoding ideas that we learned about in the previous course.

![](https://i.ibb.co/Ny7pv6J/image.png)
Remember that in this problem, the goal was to encode each face into a vector and then check how similar the faces are. 

The words encoding and embedding mean fairly similar things. So in the face recognition literature, people also use the term encoding to refer to these final vectors

In the word embeddings task, we are learning a representation for each word in our vocabulary (unlike in image encoding where we have to map each new image to some n-dimensional vector). We will discuss the algorithm in next sections.

We just saw how by replacing the one-hot vectors we're using previously with the embedding vectors, you can allow your algorithms to generalize much better, or you can learn from much less label data. Let's see few more properties of these word embeddings.

### Properties of word embeddings

One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. While reasonable analogies may not be by itself the most important NLP application, they might also help convey a sense of what these word embeddings are doing, what these word embeddings can do.

Let's see analogies with an example :

![](https://i.ibb.co/7kFSWvQ/image.png)

Let's say Andrew pose a question, "man is to woman as king is to what"? Can the algorithem automatically find the answer ? Here how it can do it.

- We have a vector E(man) that represent man, and the embedding vector for woman is E(woman).
- We have the same for queen and king: E(queen) and E(king).
- If we take E(man) - E(woman) the result is [-2, 0, 0] and E(king) - (queen) is approximatively equal to [-2, 0, 0]
- what this is capturing is that the main difference between man and woman is the gender. the same for queen and king. That is why the to substraction have the same result.

Let's formalize how we can do this in an alogithm:

we can reformulate the problem to find:
    - e<sub>Man</sub> - e<sub>Woman</sub> ≈ e<sub>King</sub> - e<sub>??</sub>
  
  - It can also be represented mathematically by:

![](https://i.ibb.co/6R6ZpzP/image.png)
the most commonly used similarity function is called cosine similarity. So in cosine similarity, you define the similarity between two vectors `u` and `v` as:

![](https://i.ibb.co/8XZVn4g/image.png)

This is basically the inner product between u and v. If u and v are very similar, their inner product will tend to be large. We called cosine similarity because this is actually the cosine of the angle between the two vectors.

We can also use the euclidian distance (thecnically it's a mesure of a dissimilarity, so we have to take the negative of it).
We can use this equation to calculate the similarities between word embeddings and on the analogy problem where `u` = e<sub>w</sub> and `v` = e<sub>king</sub> - e<sub>man</sub> + e<sub>woman</sub>

Note that the 2D representation used in the course video in made by t-SNE and that we can not rely on t-SNE to see analgie because it's a non-linear mapping, so we should not expect these types of parallelogram relationships. The image representation is not includes in my note.

### Embedding matrix

Let's start to formalize the problem of learning a good word embedding. To do that the algorithm end up learning an embedding matrix. 

Let's take an example :

![](https://i.ibb.co/fdbVK3t/image.png)
The algorithm should create a matrix E of the shape (300, 10000) in case we are extracting 300 features.

If O<sub>6257</sub> is the one hot encoding of the word **orange** of shape (10000, 1), then   
    _np.dot(`E`,O<sub>6257</sub>) = e<sub>6257</sub>_ which shape is (300, 1).
  - Generally _np.dot(`E`, O<sub>j</sub>) = e<sub>j</sub>_
  - So this give us an extract of the embeddings of a specific word.

- In the next sections, you will see that we first initialize `E` randomly and then try to learn all the parameters of this matrix.
- In practice it's not efficient to use a dot multiplication when you are trying to extract the embeddings of a specific word, instead, we will use slicing to slice a specific column. In Keras there is an embedding layer that extracts this column with no multiplication.


## Learning Word Embeddings: Word2vec & GloVe

### Learning word embeddings

We'll start to learn some concrete algorithms for learning word embeddings.

In beggining of the history of DL, researcher started with very complex algorithms hen discover then simpler algorithms still get a good result for large dataset.
We will start by seeing complex alogithm to understand how they work and to understand that the simpler algothims are not working by magic, so to understand them better.

Let's say we are building a langage model :

![](https://i.ibb.co/hVHJXzx/image.png)

The model have to predict the next word of a sentence.
The first thing we are going to do is getting e<sub>j</sub>, which is the embedding of one exact word j by doing `np.dot(`E`,o<sub>j</sub>)` for j in range all word of our sentence.

Next, we give all these values to a layer of a NN, this layer will have W1, b1 as parameters, the result of this layer goes to a softmax layer with parameters W2 and b2.

all the values tht we are inputing into the layer  are the E vectors stak vertically together. So if the size of e is (300) then the size of the input of this layer is (number of e * size of e)
For our exemple there are 6 vectors E of size (300, 1) this give us a size of (1800, 1). Here the windows size is 6 because we took all the 6 previous word. If the windows size was equal to 4, then the input size would be (1200, 1).
So we have to have a **fix historical window**. 

The plus is that using a fixed history, just means that you can deal with even arbitrarily long sentences because the input sizes are always fixed.

Here we are optimizing E matrix and layers parameters. We need to maximize the likelihood to predict the
next word given the context (previous words).

This is one of the earlier and pretty successful algorithms for learning word embeddings - for learning this matrix E.

If we want to build a langage model, it is natural to want to take the few words before the one we want to predict (a window of 4).
But there are other context :

![](https://i.ibb.co/PWdjRx5/image.png)
Suppose we have an example: "I want a glass of orange **juice** to go along with my cereal" where **juice** is our target.

  - To learn **juice**, choices of **context** are:
    1. Last 4 words.
       - We use a window of last 4 words (note that 4 is a hyperparameter), "<u>a glass of orange</u>" and try to predict the next word from it.
    2. 4 words on the left and on the right.
       - "<u>a glass of orange</u>" and "<u>to go along with</u>"
    3. Last 1 word.
       - "<u>orange</u>"
    4. Nearby 1 word.
       - "<u>glass</u>" word is near juice.
       - This is the idea of **skip grams** model. The idea is much simpler and works remarkably well. We will talk about this in the next section.
      
Researchers found that if you really want to build a _language model_, it's natural to use the last few words as a context. But if your main goal is really to learn a _word embedding_, then you can use all of these other contexts and they will result in very meaningful work embeddings as well. 

- To summarize, the language modeling problem poses a machines learning problem where you input the context (like the last four words) and predict some target words. And posing that problem allows you to learn good word embeddings.

### Word2Vec

We saw how we can learn a neural language model in order to get good word embeddings. In this part f the course, we will see the **Word2Vec** algorithm which is a simple and comfortably more efficient way to learn this types of embeddings.

In the skip-gram model, if we have this sentence "I want a glass of orange juice to go along with my cereal"

What we want to do is choosing a few context and target, we randomly pick a word to be the context word and for the target we randomly pick another word within some window. 


What we are doing here is setting up a supervised learning problem where given a context word, we are asked to predict a target that is randomly chosen within a window.
This is not an easy problem but it goal is not predict the target but it is to use this learning problem to learn good word embeddings.

![](https://i.ibb.co/qMrwmWG/image.png)

Here the detail of the model :
- we will continue to use our vocab of 10.000 words
- the basic supervised learning problem we're going to solve is that we want to learn the mapping from some Context `c`to some target `t`.
- We will get e<sub>c</sub> by `E`. o<sub>c</sub>
- Then we will take e<sub>c</sub> and feed it to a softmax layer to get `P(t|c)` which is y&#770.
- We will use the cross-entropy loss function.

the green rectangle is the overall little model , little neural network with basically looking up the embedding and then just a soft max unit. And the matrix E will have a lot of parameters, so the matrix E has parameters corresponding to all of these embedding vectors, E<sub>C</sub>. And then the softmax unit also has parameters that gives the theta T parameters but if you optimize this loss function with respect to the all of these parameters, you actually get a pretty good set of embedding vectors.


This is called the skip-gram model because is taking as input one word and then trying to predict some words skipping a few words from the left or the right side to predict what comes little bit before little bit after the context words.

It turns out that there are some problems with using this algorithm and the primary problem is the computational speed of the softmax layer. In particular to evaluate the probability we have to go over the sum of all the size of the vocabulary. 

One of the solutions for the last problem is to use "**Hierarchical softmax classifier**" which works as a tree classifier. this classifier tells you where is you target : it is in the first 5000 words or the seconds 5000 word? etc. This mean that each of the retriever nodes of the tree can be just a binding classifier.

- Here the computational cost would be log(vocab) size instead of linear in vocab size for the normal softmax layer.

![](https://i.ibb.co/7vMQXtz/image.png)

 In practice, the hierarchical software classifier can be developed so that the common words tend to be on top, whereas the less common words like durian can be buried much deeper in the tree so it doesn't use a balanced tree like the drawn one.

How to sample the context c ?

One thing you could do is just sample uniformly, at random, from your training corpus. If you have done it that way, there will be frequent words like "the, of, a, and, to, .." that can dominate other words like "orange, apple, durian,..." and we don't want that. In practice, we don't take the context uniformly random, instead there are some heuristics to balance the common words and the non-common words.

word2vec paper includes 2 ideas of learning word embeddings. One is skip-gram model and another is CBoW
(continuous bag-of-words).

### Negative Sampling

We saw how the Skip-Gram model allows us to construct a supervised learning task. So we map from context to target and how that allows us to learn a useful word embedding.

But the downside of that was the Softmax objective was slow to compute.

In this part, we'll see a modified learning problem called negative sampling that allows us to do something similar to the Skip-Gram model you saw just now, but with a much more efficient learning algorithm.

What we are going to do is create a new supervised learning problem:

![](https://i.ibb.co/WfgQF58/image.png)
The way we generated this data set is, we'll pick a context word and then pick a target word and that is the first row of this table. That gives us a positive example. 

Note that we get positive example by using the same skip-grams technique, with a fixed window that goes around.

So context, target, and then give that a label of 1. And then what we'll do is for some number of times say, k times, we're going to take the same context word and then pick random words from the dictionary, king, book, the, of, whatever comes out at random from the dictionary and label all those 0, and those will be our negative examples.

Notice that we got word "of" as a negative example although it appeared in the same sentence.

So the problem is really given a pair of words like orange and juice, do you think they appear together ? It's really to try to distinguish between these two types of distributions from which you might sample a pair of words.

Note that k is recommended to be from 5 to 20 in small datasets. For larger ones - 2 to 5.

- Now let's define the model that will learn this supervised learning problem:
  - Lets say that the context word are `c` and the word are `t` and `y` is the target.
  - We will apply the simple logistic regression model.   
  ![](https://i.ibb.co/82gdSTq/image.png)

  - The logistic regression model can be drawn like this:   
  ![enter image description here](https://i.ibb.co/nCLBxwv/image.png)
  - So we are like having 10,000 binary classification problems, and we only train k+1 classifier of them in each iteration.
- How to select negative samples:
  - We can sample according to empirical frequencies in words corpus which means according to how often different words appears. But the problem with that is that we will have more frequent words like _the, of, and..._
  - The best is to sample with this equation (according to authors):   
 
 ![](https://i.ibb.co/7y7jB5q/image.png)

### GloVe word vectors
 
 We learn about several algorithms for computing words embeddings. Another algorithm that has some momentum in the NLP community is the GloVe algorithm.

This is not used as much as the Word2Vec or the skip-gram models, but it has some enthusiasts because of its simplicity.

GloVe stands for Global vectors for word representation.
- Let's use our previous example: "I want a glass of orange juice to go along with my cereal".
- We will choose a context and a target from the choices we have mentioned in the previous sections.
- Then we will calculate this for every pair: X<sub>ct</sub> = # times `t` appears in context of `c`
- X<sub>ct</sub> = X<sub>tc</sub> if we choose a window pair, but they will not equal if we choose the previous words for example. In GloVe they use a window which means they are equal
- The model is defined like this:   
  ![](https://i.ibb.co/gRsP7vp/image.png)
- f(x) - the weighting term, used for many reasons which include:
  - The `log(0)` problem, which might occur if there are no pairs for the given target and context values.
  - Giving not too much weight for stop words like "is", "the", and "this" which occur many times.
  - Giving not too little weight for infrequent words.
- **Theta** and **e** are symmetric which helps getting the final word embedding. 

- _Conclusions on word embeddings:_
  - If this is your first try, you should try to download a pre-trained model that has been made and actually works best.
  - If you have enough data, you can try to implement one of the available algorithms.
  - Because word embeddings are very computationally expensive to train, most ML practitioners will load a pre-trained set of embeddings.
  - A final note that you can't guarantee that the axis used to represent the features will be well-aligned with what might be easily humanly interpretable axis like gender, royal, age.

## Application using Word Embeddings

### Sentiment Classification

Sentiment classification is the task of looking at a piece of text and telling if someone likes or dislikes the thing they're talking about.

![](https://i.ibb.co/7W9C5Tq/image.png)
One of the challenges with it, is that you might not have a huge labeled training data for it, but using word embedding can help getting rid of this. The common dataset sizes varies from 10,000 to 100,000 words.

A simple sentiment classification model would be like this:

![](https://i.ibb.co/RS0RZmm/image.png)
The embedding matrix may have been trained on say 100 billion words.

Number of features in word embedding is 300.
We can use sum or average given all the words then pass it to a softmax classifier. That makes this classifier works
for short or long sentences.

One of the problems with this simple model is that it ignores words order. For example "Completely lacking in good taste, good service, and good ambience" has the word good 3 times but its a negative review.

A better model uses an RNN for solving this problem:
![](https://i.ibb.co/qykC5TB/image.png)
And so if you train this algorithm, you end up with a pretty decent sentiment classification algorithm.
Also, it will generalize better even if words weren't in your dataset. For example you have the sentence "Completely
absent of good taste, good service, and good ambience", then even if the word "absent" is not in your label training
set, if it was in your 1 billion or 100 billion word corpus used to train the word embeddings, it might still get this
right and generalize much better even to words that were in the training set used to train the word embeddings but
not necessarily in the label training set that you had for specifically the sentiment classification problem.

### Debiasing word embeddings

Machine learning and AI algorithms are increasingly trusted to help with, or to make, extremely important decisions. And so we like to make sure that as much as possible that they're free of undesirable forms of bias, such as gender bias, ethnicity bias and so on.

Some horror here : 

![](https://i.ibb.co/6RkYDBg/image.png)
Andrew thinks we actually have better ideas for quickly reducing the bias in AI than for quickly reducing the bias in the human race, although it still needs a lot of work to be done.

Addressing bias in word embeddings steps:

- Idea from the paper: https://arxiv.org/abs/1607.06520

Given some word embedding, let's solve a gender bias. 

Here are the steps:
    1. Identify the direction:
       - Calculate the difference between:
         - e<sub>he</sub> - e<sub>she</sub>
         - e<sub>male</sub> - e<sub>female</sub>
         - ....
       - Choose some k differences and average them.
       - This will help you find this:   
         
       - By that we have found the bias direction which is 1D vector and the non-bias vector which is 299D vector.
    2. Neutralize: For every word that is not definitional, project to get rid of bias.
       - Babysitter and doctor need to be neutral so we project them on non-bias axis with the direction of the bias  
    
         - After that they will be equal in the term of gender.
         - To do this the authors of the paper trained a classifier to tell the words that need to be neutralized or not.
    3. Equalize pairs
       - We want each pair to have difference only in gender. Like:
         - Grandfather - Grandmother
         - He - She
         - Boy - Girl
       - We want to do this because the distance between grandfather and babysitter is bigger than babysitter and grandmother
      
       - To do that, we move grandfather and grandmother to a point where they will be in the middle of the non-bias axis.
       - There are some words you need to do this for in your steps. Number of these words is relatively small.

![enter image description here](https://i.ibb.co/MVyKYpj/image.png)

