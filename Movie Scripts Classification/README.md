# Movie Scripts Classification

   <img src=https://pbblogassets.s3.amazonaws.com/uploads/2014/12/Screenwriting-Websites-Cover-Image-865x505.jpg height ="50%"  width = "50%">

<b>Background</b>  - This notebook is a solution to [<b>Predict The Movie Genre</b>](https://analyticsindiamag.com/movie-script-classification-hackathon/) hackathon conducted by MachineHack.

<b>Problem</b> - Given movie scripts (which are long documents), we need to classify them into different genres. 

<b>Data</b> - Training data consists of script file along with the labels(Genre name is not given, just the genre id). For tests data, given the script, we need to predict the the label

<b>Approach</b> - So, we need to extract the features from the movie scripts. Since the competition organisers themselves encouraged to use Transformers, I skipped the classical techniques like Tf-Idf etc aand directly jumped to transformers

We need to create encodings for these long documents, some of them with nearly 50,000 words. But, transformers can be fed sequences whose length can be atmost 512 or 1024(generally). So, Unlike for tweets where they are truncated to 512 tokens , the same cant be done here because we'll be losing lots of information

So, I divided each script into series of sequences of equal length(512), get the sequence embeddings with transformers and then took the mean of sequence embeddings to create document embedding. Taking these document embeddings as features, 5-fold cross validation is performed using Logistic regression classifier.


I used Google Colab to work with GPUs. The major problem which I have encountered is “ CUDA: out of memory” errors . The steps which I took to solve this issue:
1)	Processing one script at a time instead of batch of scripts.
2)	Using distilled models
3)	Deleting unused variables at each step
4)	Switching of gradient calculations. 

Failed ideas – Treating each encoding as an example, with script label as its label and fitting a neural network over it. Failed because of too much noise

Other ideas worth exploring which I could not try :
1)	Using more powerful models instead of just Logistic Regression
2)	Address class imbalance with oversampling 
3)  Alternate methods to calculate document embeddings instead of just averaging sequence embeddings . But need more data, for these methods to be successful.

My most important learning from this competition – Trust your K-fold cross validation because the the split used for calculating public leaderboard scores might be skewed
