# Disaster Tweets Classifcation

<img src=https://miro.medium.com/max/1476/0*z9jqZsQ7JSTZGSZz.jpg height ="50%"  width = "50%">

<b>Background</b>  - This notebook is a solution to [<b>Disaster Tweets Classification</b>](https://www.kaggle.com/c/nlp-getting-started) competition conducted by Kaggle.

<b>Problem</b> - Given tweets, we need to classify them whether they are disaster related or not

<b>Approach</b> -  Using regex hashtags, handles and urls are  removed  from tweets.

Then tweets are tokenized (with stop words removed), features are created using classical methods like Count Vectorizer and Tf-Idf vectorizer. Finally, Logistic Regression and Multinomial Naive Bayes classifers are used to classify them
