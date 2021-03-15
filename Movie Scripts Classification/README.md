My motive in choosing the competition is to improve my understanding and application of transformer based models. So, I straightaway started with them. 
I used Google Colab to work with GPUs. The major problem which I have encountered is “ CUDA: out of memory” errors . The steps which I took to solve this issue:
1)	Processing one script at a time instead of batch of scripts.
2)	Using distilled models
3)	Deleting unused variables at each step
4)	Switching of gradient calculations. ( Even model when is in eval mode, it is still caclculating gradients. So need to use torch.no_grad())
While encoding, I used sequences of maximum possible lengths instead of individual sentences, because one of the strengths of transformer models is the self-attention mechanism. Longer sequences gives better contextual word embeddings.
Then for each script, I took the mean of sequence encodings and applied a Logistic regression on it to establish a baseline which finally turned out to be a winning solution.
My leaderboard score was too bad compared to validation scores. So, after referring some Kaggle discussions I have realized that Stratified K-fold cross validation gives better estimate of model performance. And indeed, my final score is nearly within one standard deviation of my 5-fold CV score.  
Failed ideas – Treating each encoding as an example, with script label as its label and fitting a neural network over it. Failed because of too much noise
Other ideas worth exploring which I could not try :
1)	Using more powerful models instead of just Logistic Regression
2)	Address class imbalance with oversampling (I think not doing this might be the reason of bad leaderboard scores
My most important learning from this competition – Trust your K-fold cross validation because the the split used for calculating public leaderboard scores might be skewed
