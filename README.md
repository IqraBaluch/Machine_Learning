# Machine_Learning
In the repository I will be sharing my knowledge on AI, its subsets(ML, DL, Data Science) and different algorithms used to train models.
![image](https://user-images.githubusercontent.com/57810189/116005301-c5f07800-a61f-11eb-9ec0-f99147051e95.png)

# Prerequisites
1. Python 3
2. High School mathematics
3.  16hrs+/week (2 to 2.5 hour/day)


# Paer_1 : Introduction of Machine Learning:
ML is the ability to automatically learn experiences without being forcefully programmed. The term   “Explicitly being programmed” means using if/else condition for every possible situation. In the past when ML wasn’t a subset of AI, problems were being solved by explicit programming. In general as we know that nine (09) digits make 36 possible combinations. So there is a chance that, for a 9 imaged data set we could have to make 36 conditions (if/else/elif). But thanks to Machine Learning we don’t have to work this much hard as ML learns itself and works accordingly, all we need to do is train a model and leave the entire process of learning new things and prediction on model.
# Traditional Programming VS Machine Learning (AI)
### Traditional Programming:
In Traditional Programming we used to provide rules (conditions/switches) and dataset as input and we get outputs. 
 
### Machine Learning (AI)
While in ML we provide Answers and dataset as input and get rules, than the model works according to the rules. 
 
# Machine Learning Process Step by Step
How we use ML in models? What is the flow of ML working on a problem?
 
## Step_1:  Define the objectives of the problem
•	What are we trying to predict?
For baggers the first step is to know and write down what we are trying to predict!
•	What are the target features?
If we are working on a dataset of fruits and we want to predict the rottenness of fruits in percentage (50% rotten, 60% rotten, 99.95% rotten) , so here the target features are the rottenness in fruits. 
•	What is the input data?
Input data can be anything that is available as dataset (images, texts, videos, and audios) so here we have data set of images of fruits in different conditions (fresh, not fresh, rotten and extremely rotten fruits).
•	What kind of problem we are working with??
ML problems can either be of Binary Classification, clustering or other.
Binary means 1 or 0, fresh or rotten while in clustering the data is not labeled so computer makes different groups from dataset. The fresh fruits will be in a separate group and the rotten once would be in other group.

## Step_2: Data Gathering
There are different ways to gather data and we usually prefer Kaggle.
But sometimes there are problems that have no previous data, for such problems we have to collect data ourselves. 
So here are some ways to collect data,
1. Interviews 
2. Survey and Questioners
3. Online Quiz
4. Google form 
There are many more ways, what suits your problem go with that.

## Step_3: Data Preprocessing
Data cleaning is an import part of ML, as the more data is cleaned the more model works accurately. Even if we get data from Kaggle that also required to be cleaned. Because data sets have 
. missing values
. corrupted Values
. Unnecessary data that need to be removed.
And if we have getherd our own data set by interview, google forms or any other wat that also need to be transformed into a desired format so that it is easy to work with it further. 
. Reducing dimensions of data, if we have a dataset of images, we make sure all the data that enters in the model is of the same dimension. Even if some images are dimensionally different from others so we have to do preprocessing so that any image that enters in the model turns to the same dimension, this is done for better and accurate results. 

## Step_4: Exploring Data Analysis / Visualization
After preprocessing data we analyze what useful feathers the data set has.
We need to understand the data, we need to visualize it.

## Step_5: Building ML model
Data is split into two parts, training and testing. We use 76% of data for training and remaining 33% is used for testing. 
#### MODEL:
Model is a machine learning algorithm that predicts the output by the data given to it. 

## Step_6: Model Evaluation & Optimization
Testing data is used for evaluation of model. Model is never perfect at first attempt; we need to change parameters according to results during evaluation process. 

## Step_7: Prediction
The point where model is ready to be used and we can give different new inputs to let the model predict according to its accuracy. 

## Step_8: Building a model package for production
Once the code for model is done we need to save the file, that pre trained model’s file can be used anywhere we want to use it. If we only copy the code of course we have to again train the model and do all the process from start, the accuracy changes every time the model is trained. Therefor we use the same model with the best accuracy we have saved. The way we import libraries (import pandas as pd) the same way we import our model and use it. 

# Types of Machine Learning
There are 3 main types of ML
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning 
## •	Supervised learning
In supervised learning the data is labeled.  You have images of cats and dogs and you provide labels with the images so the model knows these are the images of dog and these are cats. 
###  Types
1. Classification
2. Regression 

##### Classification
Classification means assigning each data a cetegory.

-What day is today? <br>
monday, tuesday, friday<br>
-Is it cold outside?<br>
Yes/Nope

### <a href="https://iqraanwar.medium.com/machine-learning-the-subset-of-artificial-intelligence-74cd085fae86?source=your_stories_page-------------------------------------">CLICK ON THE HERE FOR MORE DETAILS ABOUT INTRODUCTION OF ML</a>



# Part_2 : Preprocessing 
<b> Why do we need preprocessing?</b> Real world data/ raw data we get, is never clean, raw data has dummy data; it needs to be properly cleaned so that the model understands the data and does not get confused. Preprocessing allows us to purify raw data with the use of data cleaning, it is said that the more data is cleaned the more useful insights we get. For the sake of not disturbing Machine Learning’s models’ accuracy we have to first do data preprocessing. 
### Machine Learning and Data 
For Ml data is pattern, ML understands patterns from the dataset but if the data has not gone through data processing and data is not cleaned or unscaled so predictions will always be unacceptable. 
## Preprocessing Techniques 
# 1. StandardScalling
Why do we use it? It transforms continuous data to normally distributed data. 
Just a quick glance on data!! 
There are mainly two types of data

1. Qualitative/Descriptive
2. Quantitative 

<b>Quantitative data further has two types</b> 

1. Discrete data
2. Continuous data
![image](https://user-images.githubusercontent.com/57810189/120835898-61113180-c57e-11eb-8622-37f890aa7e78.png)

