
# Structuring Machine Learning Projects

## Introduction to ML strategy

### Why ML Strategy

Say that after working for some time, you have 90% accuracy for one of your application and that isn't good enough. You might then have a lot of ideas as to how to improve your system :

- Collect more data.
- Collect more diverse training set.
- Train algorithm longer with gradient descent.
- Try different optimization algorithm (e.g. Adam).
- Try bigger network.
- Try smaller network.
- Try dropout.
- Add L2 regularization.
- Change network architecture (activation functions, # of hidden units, etc.)

So, assuming you don't have six months to waste on your problem, won't it be nice if you had quick and effective ways to figure out which of all of these ideas are worth pursuing ?

This course will point you in the direction of the most promising things to try.

### Orthogonalization

Some people are very clear-eyed about what to tune in order to try to achieve one effect. This is a process we call orthogonalization.

- In orthogonalization, you have some controllers with one specific task, one can not affect the other controller's job. In the contrary you can have only one controller and this can tune several task. (Example of the TV with several  knobs VS TV with one knob to control the rotation, the position, the shape etc of an image).

For a supervised learning system to do well, you usually need to tune the knobs of your system to make sure that four things hold true : 

First, is that you usually have to make sure that you're at least doing well on the training set, then the dev set, then the test set and finally in the real word. 

So what we want is, if our algorithm don't perform well in one of these, we will find a knob to adjust it's performances.  

![orthogonalization](https://i.ibb.co/YBxN7PL/image.png)


- When training algorithm, Andrew don't use early stopping, because it's affect different knobs. 

In this course, we will identify the specific set of knobs we could use to tune our system to improve that one aspect of its performance.

## Setting up your goal

### Single number evaluation metric

The work in machine learning is often, you have an idea, you implement it try it out, and you want to know whether your idea helped.

Having a single number evaluation metric can really improve your efficiency. So it's important to set a single number evaluation metrics. 

For the cat classification example, two metrics can be the Precision and the Recall.

Remember that :
- Precision: percentage of true cats in the recognized result
- Recall: percentage of true recognition cat of the all cat predictions.

Using a precision/recall for evaluation is good in a lot of cases, but separately they don't tell you which algorithms is better. So one solution for using a single number metric is to use F1 score which combines them.

To remember the F1 Score is the "Harmonic mean" of precision and recall.

    F1 = 2 / ((1/P) + (1/R))


### Satisficing and Optimizing metric

It's not always easy to combine all the things you care about into a single row number evaluation metric.

If we care about the F1 score and the running time of our algorithm like : 

| Classifier | Accuracy (for ex:F1)   | Running time |
  | ---------- | ---- | ------------ |
  | A          | 90%  | 80 ms        |
  | B          | 92%  | 95 ms        |
  | C          | 92%  | 1,500 ms     |

What we can do in this case is,  setting one as the optimizing metric that you want to do as well as possible on and one or more as satisfying metrics were you'll be satisfied.

For our example, that would be : 

    Maximize F1 # optimizing metric
    subject to running time < 100ms # satisfying metric

B is the best choice.

### Train/dev/test distributions

The way you set up your training dev, or development sets and test sets, can have a huge impact on how rapidly you can make progress on building machine learning application.

Here some guidelines :

- Make sure that your dev set and your test set come from the same distribution. 
- Choose dev set and test set to reflect data you expect to get in the future and consider important to do well on.
- setting up the dev set, as well as the validation metric, is really defining what target you want to aim at.
- The way you choose your training set will affect how well you can actually hit that target.


### Size of the dev and test sets

An old way of splitting the data was 70% training, 30% test or 60% training, 20% dev, 20% test. The old way was valid for a number of examples ~ <100000

In the modern deep learning if you have a million or more examples a reasonable split would be 98% training, 1% dev, 1% test.

Remember the purpose of your test set is that, after you finish developing a system, the test set helps evaluate how good your final system is. The guideline is, to set your test set to big enough to give high confidence in the overall performance of your system.


- if all you care about is having some data that you train on, and having some data to tune to, and you're just going to not worry too much about how it was actually doing, it will be healthy and just call the train dev set and acknowledge that you have no test set. But a test set is very important, so make sure you use one.

### When to change dev/test sets and metrics

If we have two algorithms, one (A) with the best metric (the error is let's say 3%)  and the other ( B) with 5% error. 

If we found out that for example :
- A don't fit the user cases, because they use different images than what we use in the dev/set.
- A return somethings we want to avoid (like porn image in image classification) 

Then in this case, we want and need to change our metric or/and our dev/test set. 

If our old metric is :

    OldMetric = (1/m) * sum(y_pred[i] != y[i] ,m)

Where m is the number of Dev set items.

the new metric would be :

     (1/sum(w[i])) * sum(w[i] * (y_pred[i] != y[i]) ,m)

where:

    w[i] = 1 if x[i] is not porn
    w[i] = 10 if x[i] is porn

This is actually an example of an orthogonalization where you should take a machine learning problem and break it into distinct steps:

i. Figure out how to define a metric that captures what you want to do - place the target.
ii. Worry about how to actually do well on this metric - how to aim/shoot accurately at the target.


**So, if doing well on your metric + dev/test set doesn't correspond to doing well in your application, change your metric and/or dev/test set.**

### Why human-level performance?

In the last few years, a lot more machine learning teams have been talking about comparing the machine learning systems to human level performance.
There are two main reasons why : 

- First is that because of advances in deep learning, machine learning algorithms are suddenly working much better. It's much feasible for algorithms to actually become competitive with human-level performance.

- Second, it turns out that the workflow of designing and building a machine learning system, the workflow is much more efficient when you're trying to do something that humans can also do.

So it becomes natural to talk about comparing, or trying to mimic human-level performance.

After an algorithm reaches the human level performance the progress and accuracy slow down. And over time, the performance approaches but never surpasses some theoretical limit, which is called the Bayes optimal error.

Note that there isn't much error range between human-level error and Bayes optimal error.

For tasks that humans are quite good at, (and this includes looking at pictures and recognizing things, or listening to audio, or reading language, etc.) so long as your machine learning algorithm is still worse than the human, you can :

- get labeled data from humans
- Gain insight from manual error analysis: why did a person get it right?
- Better analysis of bias/variance.

### Avoidable bias


- Suppose that the cat classification algorithm gives these results:

  | Humans             | 1%   | 7.5% |
  | ------------------ | ---- | ---- |
  | **Training error** | 8%   | 8%   |
  | **Dev Error**      | 10%  | 10%  |
  
  - In the left example, because the human level error is 1% then we have to focus on the **bias**.
  - In the right example, because the human level error is 7.5% then we have to focus on the **variance**.
  - The human-level error as a proxy (estimate) for Bayes optimal error. Bayes optimal error is always less (better), but human-level in most cases is not far from it.
  - You can't do better then Bayes error unless you are overfitting.
  - Avoidable bias = Training error - Human (Bayes) error`
  - Variance = Dev error - Training error`

### Understanding human-level performance

- You might have multiple human-level performances based on the human experience. Then you choose the human-level performance (proxy for Bayes error) that is more suitable for the system you're trying to build.

Summary of bias/variance with human-level performance:
- human-level error (proxy for Bayes error)
	- Calculate avoidable bias = training error - human-level error
	- If avoidable bias difference is the bigger, then it's bias problem and you should use a strategy for bias resolving.
- training error
	- Calculate variance = dev error - training error
	- If variance difference is bigger, then you should use a strategy for variance resolving.

This allows you to more quickly make decisions as to whether you should focus on trying to reduce a bias or trying to reduce the variance of your algorithm.

- Improving deep learning algorithms is harder once you reach a human-level performance.

### Surpassing human-level performance

there are many problems where machine learning significantly surpasses human-level performance : 

- Online advertising
- Product recommendation
- Loan approval

All these examples are on structural data.

It's harder for machines to surpass human-level performance in natural perception task. But there are already some systems that achieved : medical task, speech reco...

### Improving your model performance

Just summary of this week courses:

The two fundamental assumptions of supervised learning:

i. You can fit the training set pretty well. This is roughly saying that you can achieve low avoidable bias.
ii. The training set performance generalizes pretty well to the dev/test set. This is roughly saying that variance is not too bad.

To improve your deep learning supervised system follow these guidelines:

i. Look at the difference between human level error and the training error - avoidable bias.

ii. Look at the difference between the dev/test set and training set error - Variance.

iii. If avoidable bias is large you have these options:

- Train bigger model.
- Train longer/better optimization algorithm (like Momentum, RMSprop, Adam).
- Find better NN architecture/hyperparameters search.

iv. If variance is large you have these options:

- Get more training data.
- Regularization (L2, Dropout, data augmentation).
- Find better NN architecture/hyperparameters search.

## Error Analysis

### Carrying out error analysis

If you're trying to get a learning algorithm to do a task that humans can do and it's not yet at the performance of a human, then manually examining mistakes that your algorithm is making, can give you insights into what to do next. This process is called **error analysis**.


In the cat classification example, if you have 10% error on your dev set and you want to decrease the error.

You discovered that some of the mislabeled data are dog pictures that look like cats. 
Should you try to make your cat classifier do better on dogs (this could take some weeks)?

Error analysis approach:

- Get 100 mislabeled dev set examples at random.
- Count up how many are dogs.
- if 5 of 100 are dogs then training your classifier to do better on dogs will decrease your error up to 9.5% (called ceiling), which can be too little.
- if 50 of 100 are dogs then you could decrease your error up to 5%, which is reasonable and you should work on that.


Error analysis helps you to analyze the error before taking an action that could take lot of
time with no need.

Sometimes, you can evaluate multiple error analysis ideas in parallel and choose the best idea. Create a spreadsheet to do that and decide: 


![error_analysis](https://i.ibb.co/hm1cpk9/image.png)

this quick counting procedure, which you can often do in, at most, small numbers of hours. Can really help you make much better prioritization decisions, and understand how promising different approaches are to work on.


### Cleaning up incorrectly labeled data


What if you going through your data and you find that some of these output labels Y are incorrect ?

It turns out that deep learning algorithms are quite robust to random errors in the training set. So long as your errors or your incorrectly labeled examples are not too far from random, then it's probably okay to just leave the errors as they are and not spend too much time fixing them.

If you want to check for mislabeled data in dev/test set, you should also try error analysis with the mislabeled column:

![misslabeed](https://i.ibb.co/8mXdRQv/image.png)
Then:
- If overall dev set error: 10%
- Then errors due to incorrect data: 0.6%
- Then errors due to other causes: 9.4%
- Then you should focus on the 9.4% error rather than the incorrect data.


Consider these guidelines while correcting the dev/test mislabeled examples:
- Apply the same process to your dev and test sets to make sure they continue to come from the same distribution.
- Consider examining examples your algorithm got right as well as ones it got wrong. (Not always done if you reached a good accuracy)
- Train and (dev/test) data may now come from a slightly different distributions.
- It's very important to have dev and test sets to come from the same distribution. But it could be OK for a train set to come from slightly other distribution.


### Build your first system quickly, then iterate

If you're working on a brand new machine learning application, one of the piece of advice Andrew often give people is that you should build your first system quickly and then iterate.

You  need to :

- Set up dev/test set and metrics 
- Build initial system quickly (it can be a quick and dirty implementation, don't overthink it)
- Use Bias/Variance analysis & Error analysis to prioritize next steps.

## Mismatched training and dev/test set
        
### Training and testing on different distributions

In a deep learning era, more and more teams are now training on data that comes from a different distribution than your dev and test sets.And there's some best practices for dealing with it.

If you have additional data with not the same distribution then the training, test and dev set, then here are some strategies to follow :


![enter image description here](https://i.ibb.co/zZdthL6/image.png)
- Option one (not recommended): shuffle all the data together and extract randomly training and dev/test sets.
	- Advantages: all the sets now come from the same distribution.
	- Disadvantages: the other (real world) distribution that was in the dev/test sets will occur less in the new dev/test sets and that might be not what you want to achieve.

-Option two: take some of the dev/test set examples and add them to the training set.
	- Advantages: the distribution you care about is your target now.
	- Disadvantage: the distributions in training and dev/test sets are now different. But you will get a better performance over a long time.

### Bias and Variance with mismatched data distributions

Estimating the bias and variance of your learning algorithm really helps you prioritize what to work on next. But the way you analyze bias and variance changes when your training set comes from a different distribution than your dev and test sets.

![enter image description here](https://i.ibb.co/fx3Dwvz/image.png)

i. Human-level error (proxy for Bayes error)
ii. Train error
- Calculate avoidable bias = training error - human level error
- If the difference is big then its Avoidable bias problem then you should use a strategy for high bias.
iii. Train-dev error
- Calculate variance = training-dev error - training error
- If the difference is big then its high variance problem then you should use a strategy for solving it.
iv. Dev error
- Calculate data mismatch = dev error - train-dev error
- If difference is much bigger then train-dev error its Data mismatch problem.
v. Test error
- Calculate degree of overfitting to dev set = test error - dev error
- Is the difference is big (positive) then maybe you need to find a bigger dev set (dev set and test set come from the same distribution, so the only way for there to be a huge gap here, for it to do much better on the dev set than the test set, is if you somehow managed to overfit the dev set).

### Addressing data mismatch

what if you perform error analysis and conclude that data mismatch is a huge source of error, how do you go about addressing that? It turns out that unfortunately there are super systematic ways to address data mismatch, but there are a few things you can try that could help

There are completely systematic solutions to this, but let's look at some things you could try:

1. Carry out manual error analysis to try to understand the difference between training and dev/test sets.
2. Make training data more similar, or collect more data similar to dev/test sets.

If your goal is to make the training data more similar to your dev set one of the techniques you can use Artificial data synthesis that can help you make more training data.

- Combine some of your training data with something that can convert it to the dev/test set distribution.

Examples:
	a. Combine normal audio with car noise to get audio with car noise example.
	b. Generate cars using 3D graphics in a car classification example.
	
- Be cautious and bear in mind whether or not you might be accidentally simulating data only from a tiny subset of the space of all possible examples because your NN might overfit these generated data (like particular car noise or a particular design of 3D graphics cars).

##  Learning from multiple tasks

### Transfer learning

One of the most powerful ideas in deep learning is that sometimes you can take knowledge the neural network has learned from one task and apply that knowledge to a separate task. This is called Transfer learning.

For example, you have trained a cat classifier with a lot of data, you can use the part of the trained NN it to solve x-ray classification problem.

To do transfer learning, delete the last layer of NN and it's weights and:

	i. Option 1: if you have a small data set - keep all the other weights as a fixed weights. Add a new last layer(-s) and initialize the new layer weights and feed the new data to the NN and learn the new weights.
	ii. Option 2: if you have enough data you can retrain all the weights.
	
Option 1 and 2 are called fine-tuning and training on task A called pretraining.

When transfer learning make sense:
- Task A and B have the same input X (e.g. image, audio).
- You have a lot of data for the task A you are transferring from and relatively less data for the task B your transferring to.
- Low level features from task A could be helpful for learning task B.

### Multi-task learning

Whereas in transfer learning, you have a sequential process where you learn from task A and then transfer that to task B. In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these task helps hopefully all of the other task. Let's look at an example.


- Whereas in transfer learning, you have a sequential process where you learn from task A and then transfer that to task B. In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these tasks helps hopefully all of the other tasks. 
- Example:
  - You want to build an object recognition system that detects pedestrians, cars, stop signs, and traffic lights (image has multiple labels).
  - Then Y shape will be `(4,m)` because we have 4 classes and each one is a binary one.
  - Then   
  `Cost = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j))), i = 1..m, j = 1..4`, where   
  `L = - y(i)_j * log(y_hat(i)_j) - (1 - y(i)_j) * log(1 - y_hat(i)_j)`
- In the last example you could have trained 4 neural networks separately but if some of the earlier features in neural network can be shared between these different types of objects, then you find that training one neural network to do four things results in better performance than training 4 completely separate neural networks to do the four tasks separately. 
- Multi-task learning will also work if y isn't complete for some labels. For example:
  ```
  Y = [1 ? 1 ...]
      [0 0 1 ...]
      [? 1 ? ...]
  ```
  - And in this case it will do good with the missing data, just the loss function will be different:   
    `Loss = (1/m) * sum(sum(L(y_hat(i)_j, y(i)_j) for all j which y(i)_j != ?))`
- Multi-task learning makes sense:
  1. Training on a set of tasks that could benefit from having shared lower-level features.
  2. Usually, amount of data you have for each task is quite similar.
  3. Can train a big enough network to do well on all the tasks.
- If you can train a big enough NN, the performance of the multi-task learning compared to splitting the tasks is better.
- Today transfer learning is used more often than multi-task learning.

## End-to-end Deep Learning

### What is end-to-end deep learning?

One of the most exciting recent developments in deep learning, has been the rise of end-to-end deep learning. So what is the end-to-end learning? Briefly, there have been some data processing systems, or learning systems that require multiple stages of processing.
And what end-to-end deep learning does, is it can take all those multiple stages, and replace it usually with just a single neural network.

Example 1: Speech recognition system

![ete](https://i.ibb.co/znNnLrV/image.png)

End-to-end deep learning gives data more freedom, it might not use phonemes when training!
To build the end-to-end deep learning system that works well, we need a big dataset (more data then in non end-to-end system). If we have a small dataset the ordinary implementation could work just fine.

Example 2: Face recognition system:

![enter image description here](https://i.ibb.co/k8HD7Dc/image.png)
- In practice, the best approach is the second one for now.
- In the second implementation, it's a two steps approach where both parts are implemented using deep learning.
- Its working well because it's harder to get a lot of pictures with people in front of the camera than getting faces of people and compare them.
- In the second implementation at the last step, the NN takes two faces as an input and outputs if the two faces are the same person or not.

### Whether to use end-to-end deep learning

Pros of end-to-end deep learning:
- Let the data speak. By having a pure machine learning approach, your NN learning input from X to Y may be more able to capture whatever statistics are in the data, rather than being forced to reflect human preconceptions.
- Less hand-designing of components needed.

Cons of end-to-end deep learning:
- May need a large amount of data.
- Excludes potentially useful hand-design components (it helps more on the smaller dataset).

Applying end-to-end deep learning:
- Key question: Do you have sufficient data to learn a function of the complexity needed to map x to y?
- Use ML/DL to learn some individual components.
- When applying supervised learning you should carefully choose what types of X to Y mappings you want to learn depending on what task you can get data for.


