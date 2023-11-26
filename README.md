# Propaganda Data Generation

Generating the Propaganda unlabeled and labeled dataset from the Moderation dataset published by **X**.

## Introduction

The [X](x.com) platform frequently releases suspended propagandist users' data publicly for research studies. Those users violate the transparency policy of **X** by participating in targeted propaganda. We utilized this dataset from [Moderation Research](https://transparency.twitter.com/en/reports/moderation-research.html) website by **X** to extract a representative sample of the 2019 KSA dataset to work with in **weakly supervised propaganda detection** project.

## Data Preparation

The original dataset is around 23M tweets, containing all types of tweets (tweets, quotes, and replies) written mostly in Arabic with a minority written in different languages. We used only the tweets and removed all quotes and replies. Also, we removed all non-Arabic tweets and removed all the tweets that have likes and/or replies as missing values. If a tweet has a number of likes missing, not zero, this means that the tweet and its data might be wrong. The detailed steps are described below with the name of the notebooks.

- [Loading the data](./loading_data.ipynb): The dataset from **X** is provided as 9 CSV files. We loaded the files, 19 GBs combined, and combined them in one [Pandas](https://pandas.pydata.org/) DataFrame and saved it locally in a Pickle file.
- [Data exploration](./data_exploration.ipynb): We, then, explored the dataset and removed all quotes and replies, all non-Arabic tweets, and all like and replies null values. At this point, the resulting dataset is around 14.8M tweets.
- [Text processing](./clustering.ipynb): Processing the raw tweets is a preliminary step before clustering. We do this in the clustering notebook but it's a vital preprocessing step. We removed all links, user mentions, hashtag symbols, emojis, diacritization, extra white spaces, and new line characters. We maintained only the text of the hashtag as it may have important information.

## Representative Sampling

We seek to create a relatively small dataset that represents the 14.8M large one. For that reason, we clustered the dataset and extracted a small sample (4000 tweets) from each cluster. The choice of 4000 is to fulfill the requirements by the weak-supervision techniques used later in the detection project in case a contributor needs to work on tweets from a specific cluster. The detailed steps are described below with the name of the notebooks.

- [Clustering](./clustering.ipynb): We utilized the [FastText](https://github.com/facebookresearch/fastText) framework to train a 100-dimensional skip-gram [Word2Vec](https://jalammar.github.io/illustrated-word2vec/) model on the texts of the tweets to be used later as a text encoder. We then clustered the data using the [MiniBatchKMeans](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans) to speed up the process given the size of the dataset. The [elbow method](<https://en.wikipedia.org/wiki/Elbow_method_(clustering)>) was used to pick the number of clusters to be **14**.
- [Sampling](./sampling.ipynb): We, then, randomly picked 4000 tweets form each cluster to construct the unlabeled dataset. From the 4000 we extracted from each cluster, we further sample 150 tweets to construct the sample to be sent to Subject Matter Experts (SMEs) for labeling.

## Label Cleansing

With the help of SMEs to label the 2100 tweets as **Propaganda** and **Transparent** based on the [techniques](https://propaganda.qcri.org/annotations/definitions.html) reported by [Qatar Computing Research Institute](https://www.hbku.edu.qa/en/qcri) in the [Fine-Grained Analysis of Propaganda in News Article](https://aclanthology.org/D19-1565/) paper, we started labeling the first 500 tweets in an effort to transform the propaganda techniques from the specific-domain (_news_) to the general domain. We utilized the [CleanLab](https://github.com/cleanlab/cleanlab) framework to detect issues in the labeling. while using only the first 500 tweets, we searched for errors in the concept of propaganda techniques and human errors in the labeling. After multiple iterations, we and the SMEs managed to refine the techniques to work well on the tweets. After that, we applied the refined techniques to the entire dataset and started label cleansing once again to fix the human errors. The detailed steps are described below with the name of the notebooks.

- [Shallow label issues detection](./labeling_issues_detection.ipynb): We used a simple [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) model and trained it with the first 500 tweets. We encoded (converted the tweets from text to vectors) using the [pre-trained FastText Arabic Word2Vec model](https://fasttext.cc/docs/en/crawl-vectors.html). The model was later used along with the **CLeanLab** framework to detect the label issues. After fixing the errors in the propaganda techniques concept, we applied the same process multiple times on the entire 2100 tweets to detect and fix the human errors.
- [Transformer model training](./transformer_model_training.ipynb): To further test a real-world model on the dataset, we trained an [AraBERT](https://github.com/aub-mind/arabert) text classifier model to classify the tweets as **Propaganda** or **Transparent**. This model will be used, only, by the **CleanLab** framework to detect the label issues.
- [Transformer model label issues detection](./labeling_issues_detection_tf.ipynb): After training the transformer model, we used it along with the **CleanLab** framework to detect the label issues in the dataset. This was the final step to make sure the concepts were understandable by a large-size model. Lastly, the cleaned dataset was saved to be used in the main **Weakly Supervised Propaganda Detection** Project.

## Notes

- We made sure that the training of the transformer model didn't overfit the dataset by training only for two epochs.
- The original dataset is available by the **X** platform Moderation Research and can accessed by email.
- The contributions of this project are to extract a relatively small representative sample of the large dataset using clustering and adapting the propaganda techniques in a domain-independent scenario.
