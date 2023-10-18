# Assignment 1 Report

## Part 1 - Language identification with linear classification

### Tasks

1. Have a quick peek at the training data, looking at a couple of texts from different languages. Do you notice anything that might be challenging for the classification
   
   **Answer**: Several languages in the Latin family use the same alphabet, and the words they form are extremely similar.

2. How many instances per label are there in the training and test set? 
   
   **Answer**: 500 instances per label in the training set and 500 instances per label in the test set.

3. Do you think this is a balanced dataset? 
   
   **Answer**: To some extent, yes, each language has the same number of instances. However, the training set should contain more instances than the test set.

4. Do you think the train/test split is appropriate? 
   
   **Answer**: No. Training set should have more instances in each language than the test set. Like 4:1. Here it's 1:1.

5. If not, please rearrange the data in a more appropriate way.
   
   **Answer**: (Have done in notebook) Here is the approach we use: Firstly, we concat train_df with test_df to get a whole data frame. Each label has 1000 instances now. Secondly, for each label, we select 800 samples as new training set, and remaining 200 samples as new test set. Each language has 800 instances in the new training set and 200 instances in the new test set, where the 'training size: test size = 8:2'.

6. Get a subset of the train/test data that includes English, German, Dutch, Danish, Swedish and Norwegian, plus 20 additional languages of your choice (the labels can be found in the file labels.csv)
   
   **Answer**: (Have done in notebook) English: eng, German: deu, Dutch: nld, Danish: dan, Swedish: swe, Norwegian: nob, Japanese: jpn, and we randomly select 20 other languages in the remaining except for those ones above.

7. With the following code, we wanted to encode the labels, however, our cat was walking on the keyboard and some of it got changed. Can you fix it?
   
   **Answer**: 
   
   original code:

![](https://lh6.googleusercontent.com/0ZoDG5X_7zGPRuHSTP4FV2jn7Ar1nHJuLUAkfTm_8n90VfiFZ38vA9v47RZ-KEEy2V2iNx4BGUNV9xDybTQPdV2sct1iCaluFPnC85VlxXZZRcsR_RQaPByZLcv1UqcbraapYJjyE4TNDcuAJMYkETY)

        modified code:

![](https://lh5.googleusercontent.com/M9pAlpmN4RCjRCSTE62B1okNGClZN3BWx4d7--JOAy8v_f0e-rkUn-2pfdrTj0oos9FrLBYwre-pB4wzJ-MIaLdF1d53FZJlSUGlz09PLAAmVI1LLi50M1iFTZ7e3qPD1-7rhtVcMmGwTXkj3PIEdJk)

Reason: We can only fit the training data but not the test data. We can first fit the training data and then transform the training data and test data. So the code "le_fitted.fit(test_df['label'])" is fixed. 

### Pipeline

We use Pipeline from sklearn for more conveniently running the experiments. We use TfidfVectorizer to replace CountVectorizer and TfidfTransformer. There are two reasons: 

1. Due to Eli5, if we use TfidfTransformer there would be errors. 

2. This is also more clear and intuitive

Our pipeline is like:

<img title="" src="file:///Users/nanfangwuyu/Library/Application%20Support/marktext/images/2023-10-15-22-54-46-image.png" alt="" width="425">

Q: **What other features could you use to determine the language? Please include additional linguistic features to your machine learning model for this task.**

A: We apply ngram_range: (1, 2) to replace the original (1, 1), which means unigrams and bigrams are both used. We extend the feature space than the initial param setting. 

After vectorizing, we use LogisticRegression() as our model here. The cross validation score after training is [0.96125 , 0.96013889, 0.96333333]. 

### Hyperparameters

We use sklearn’s GridSearchCV to do cross validation to find the optimal hyperparameters. 

<img src="file:///Users/nanfangwuyu/Library/Application%20Support/marktext/images/2023-10-15-23-01-25-image.png" title="" alt="" width="532">

The hyperparameters we want to search for are:

1. regularization: we try ‘L1’ and ‘L2’

2. Solver: we try ‘liblinear’ and ‘saga’

3. Vectorizer: we try ‘unigram’ or ‘unigrams and bigrams’, plus adding ‘L1’ and ‘L2’ normalizations

The best-performing model’s hyperparameter combination:

>  'LR__penalty': 'l2',
> 
>  'LR__solver': 'saga',
> 
>  'TfidfVect__ngram_range': (1, 1),
> 
>  'TfidfVect__norm': 'l2'

Q: **What is the advantage of grid search cross-validation?**

A: GridSearchCV tries all the combinations of the values passed in the dictionary and evaluates the model for each combination using the Cross-Validation method. Hence after using this function we can get accuracy/loss for every combination of hyperparameters and we can choose the one with the best performance.

#### Best Model and Error Analysis

Then we train our best-performing model on the training data and get confusion matrix based on the predictions and test data. Here is our discovery: 

**Diagonal Elements**: In the confusion matrix, from the top-left to the bottom-right elements, all of these diagonal elements are close to 200, which is the total number of samples of each label in the test set. This means the true positive values for each label is relatively high, indicating that the classifier performs well on most classes. 

**Non-Diagonal Elements**: However, there are also non-diagonal elements, which are errors (wrong predictions). The maximum number a class is misclassified is 15, which is the number 'ang' texts are misclassified as 'eng' texts. Another big number is 14, which shows the count of 'pcd' samples that are misclassified as 'eng' texts. To be noticed, we can find that there are also some misclassifications from 'eng' to 'ang' or 'pcd' languages, though a bit fewer. These results indicate that there are some confusion between these pairs of languages.

According to the confusion matrix, we can calculate the precision, recall and f1 score for each language as shown below. 

![](https://lh6.googleusercontent.com/3_J9qGZN-La19IyqzTg4nQ03ysfN2GJC34oHlJVRk0_4yTznP7Gpm6e_B_m6Qnz3bnqQwewZB8bEsWHalrUFk8-pSB5z3dKw7RYODBPV3qQE5rDc1uCNwMVCmrUg6kkIzwXRQS_LSJxGOTV3T2lWsPY)

### Feature Importance Table for the Top Ten Features

![](https://lh3.googleusercontent.com/_JEgLzGPpvpvoXZzDCf-pfl7H_fyILXmL1LbNbVIyVGeWOxTuwDPfvmPciFubXdPdFGCLAxIGQ78UoVpqcYoohgceoymr6fXpovZkQabimhG2AAZLg7ll8RYtAaq2_mtgIC28tZiOAzaIjOO75Jd1zI)

**What is more important, extra features or the outputs of the vectorizer?** 

For vectorizer, outputs from vectorizer are the primary source of information for language classification, they capture the linguistic characteristics and patterns of each language, so they play a crucial role in achieving good accuracy.

For extra features, they can vary based on their relevance to the classification, and the source they come from. Based on the table, we can see there are more than 10k positive features and over 240k negative features used to predict the correct class. Obviously, extra features are also important. 

We think they are both important in this task. To say which is better in other tasks, it depends on the specific language classification task and the relevance of additional data. 

### Ablation Study

Q: **How does the ablation affect the performance of the classifier?**

A: We set the number of characters per instance in the training set (1. All characters, 2. 500 characters, 3. 100 characters, 4. 10 characters (added).  

**Case 1:**

The two languages for which the classifier worked best are: 'bod', 'nav'. 

> Accuracy when length is original: 1.0 
> 
> Accuracy when length is 500: 1.0 
> 
> Accuracy when length is 100: 1.0 
> 
> Accuracy when length is 10: 1.0

As a result, the ablation has no influence on the accuracy. This is probably because these two languages ( 'nav' and 'bod' ) use completely different vocabulary, they are very easy to distinguish by the best model. In this case, length of the text is slightly influential.

**Case 2:**

To see if the accuracy will decrease in other conditions, we choose two languages like 'nob' and 'swe'. 

> Accuracy when length is original: 1.0 
> 
> Accuracy when length is 500: 1.0 
> 
> Accuracy when length is 100: 1.0 
> 
> Accuracy when length is 10: 0.9225

There is still little influence on accuracy when length is not very small. When the length reduces to 10, there is an obvious drop in accuracy. 

To summarize it, the ablation barely affect the performance of the classifier.

## Part 2 - Your first Neural Network

### Tasks

> 1. Please use again the train/test data that includes English, German, Dutch, Danish, Swedish and Norwegian, plus 20 additional languages of your choice (the labels can be found in the file labels.csv) and adjust the train/test split if needed
> 
> 2. Use your adjusted code to encode the labels here
> 
> 3. In the following, you can find a small (almost) working example of a neural network. Unfortunately, again, the cat messed up some of the code. Please fix the code such that it is executable.
>    
>    (All have been done in notebook)

### Improve the Neural Network

First, we use the default hyperparameters. After 500 epochs, the validation accuracy is still under 80%. The result is as followed:

Training accuracy: 0.8454807692307692

Testing accuracy: 0.24826923076923077

The difference between training accuracy and testing accuracy is quite large, which means that there may be an overfitting in the training data.

To improve the model, we use the GridSearchCV to find the best parameters.

We make a pipeline as our model and use 'to_float32' to transform the output of the vectorizer to float 32, else there will be an error.

<img src="https://lh5.googleusercontent.com/o7Xm9iGsKoDzNqHIpvaU8Oo7hFp7JomJ4B30WpeAToX_KY_78PuZNRYhTK6oGOPPzVdfADBR8AWYgTS3Vujhumkj1MzhNRWm3WFWxuQqhqf-iKiHIRPFIgNjFQjyy3FbD2xBp3h0f2ivr_qMkefru4o" title="" alt="" width="405">

Except the parameters we fixed (max_epochs and CV max_features), the parameters we want to find best include 5 requirements in Exercise 1 document.

<img src="https://lh6.googleusercontent.com/mk3fs1pHeGyZgojuEbVqnKZptB-H_OTTNULs2uG3Eqhr6CheNDhk4HxJDHFXsS0vdkbIwqK_R91LaKSlVmdEVNAT1dkp279Tb1BMauwcpFmplDMedg_4sroBv9mmPW1lhkP8QcuuICTPhrT788X14Ws" title="" alt="" width="502">

We use 3 cross-validation sets. For saving GridSearchCV time and improving the efficiency, we sample 600 instances from the sub training set, and make it a new dataset. 

The best parameters are:

>  'CV__ngram_range': (1, 1),
> 
>  'NN__callbacks__EarlyStopping__patience': 20,
> 
>  'NN__module__nonlin': LeakyReLU(negative_slope=0.01),
> 
>  'NN__module__num_units': 300,
> 
>  'NN__optimizer__lr': 0.1

Using the params we got from GridSearchCV, we explore if increasing the number of features will make the accuracy higher. The parameter is both max_features for CV and in_features for NN (they should be the same). 

We test features in [100, 200, 500, 5000]. 

As we increase the number, the model performs better.

Result: 

> Model with features = 100 has test accuracy around 0.76 < 0.8
> 
> Model with features = 200, 500, 5000 has test accuracy around 0.83, 0.89, 0.96, all higher than 0.8.

So we have obtained higher performance, because we:

1. Use GridSearchCV to find the best performing hyperparameters. For example, setting the earlystopping will reduce the overfitting risk. 

2. We increase the number of max features the vectorizer can extract and this provides the model with more feature information for classification. 
