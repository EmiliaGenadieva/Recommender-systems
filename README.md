# Recommender-systems
Predict a movie genre based on movie description. It is fancy to see all kinds of recommender systems. This repo shows a similar example, the data is a movies dataset. The code is full with charts, comments and explanation of the process.
# Description
### Mission objective:

For this challenge, explored are different NLP methods:</br>
See example: [TfIdVectoriserRecommendations](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/TfIdVectoriserRecommendations.ipynb)
* Fine-tune a transformer model (BERT) to classify movies according to their genre based on their plot.</br>
[BERT](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/BERT.ipynb)
* Use spacy to implement one extra feature, for example key-word extraction, sentiment analysis, or text summarization.</br>
[key-word extraction](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/key_word_extraction.ipynb)
## The data:
* [MovieLens](https://grouplens.org/datasets/movielens/). A subset of this data can be found in [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

# Usage

* Explanation how to preprocess data.</br>
[preprocess file](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/preprocessing.ipynb)
* Explanation of what are the main steps to fine-tune a transformer model and how they work (i.e. encoding, picking an optmizer and loss function).</br>
The Bert models expect encoded tensor format.
AutoModel and AutoTokenizer will take the appropriate tokenizer class and encode the input. Espectially we can specify the max_length of our sentenses. It is defined in the following way:
> padding="max_length", max_length=8
</br>
A longer explanation of this functionality: 
When we initialise the transformer with padding = True, than the tokeniser will take the longest item(sentences) from out input and will append to the shorter instances 1's. This will take all the available data we have, for the shorter input the 1's will have no effect for the calculation of our result. The following is example for the definition of the optimizer, metrics and the loss:

> optimizer=tf.keras.optimizers.Adam(learning_rate=0.007)</br>
metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)</br>
loss=tf.keras.losses.SparseCategoricalCrossentropy()</br>
* Appropiate metrics to evaluate the results of approach.</br>
Mostly used for evaluation was accuracy and mean squared error.

# Improvements
See the project in production, deployed and ready to be explored. Many images of films and web layout are still needed, the current version is only limited to the most essential features, to show the power of neural network models. 

# Time frame
- Repository: `Recommender systems`
- Type of Challenge: `Learning`
- Duration: `8 days`
- Development Deadline: `25/04/2023 4:30 PM`
- Repo Deadline: `26/04/2023 12:30 PM`
- Repo Presentations: `26/04/2023 1:30PM`
- Challenge: Individual (or Team)
# Tutorials used about recommendation systems:

* [Recommender systems in Python ](https://www.datacamp.com/tutorial/recommender-systems-python)
