# Recommender-systems
Predict a movie genre based on movie description. It is fancy to see all kinds of recommender systems. This repo shows a similar example, the data is a movies dataset. The code is full with charts, comments and explanation of the process.
# Description
### Mission objective:

For this challenge, explored are different NLP methods:</br>
See example: [TfIdVectoriserRecommendations](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/TfIdVectoriserRecommendations.ipynb)
* Fine-tune a transformer model (BERT) to classify movies according to their genre based on their plot.</br>
[BERT](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/BERT_2.ipynb)
* Use spacy to implement one extra feature, for example key-word extraction, sentiment analysis, or text summarization.</br>
[key-word extraction](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/key_word_extraction.ipynb)
## The data:
* IMDB public dataset that is used for this purpose: [IMDB dataset](https://www.imdb.com/interfaces/) 
* [MovieLens](https://grouplens.org/datasets/movielens/). A subset of this data can be found in [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
The 2 datasets have different data structure. The column names are also different. By the larger public dataset we find that the there is possibility to explore sentiment analysis based on positive and negative reviews.</br>
By the smaller dataset we find that there are more options: We can investigate for the director of the movie, the actors, the ratings. There is a description and title of the movie. We find also the genre of the movie.</br>
Some </br>

# Usage

### How to preprocess data ?</br>
Notebook [preprocess file](https://github.com/EmiliaGenadieva/Recommender-systems/blob/main/notebooks/preprocessing.ipynb)</br>
It is very important to clean the data from unnecessary text characters, digits, dollar signs, single or double brackets, quotes. All these values will not help for the right prediction of the text. </br>
There are different ways to clean text data. One can use regular expressions. The library ‘Spacy’ offers functions to clean the data from digits, extra spaces and other less important text values for our model.</br>
The pandas library also has different methods to slice and dice text data. A data frame object can easily be divided to chunks of the desired format. While preparing the data file to the correct format, one needs to find the value and update the corresponding place in the new data frame.
Seen that so many preprocessing steps are needed, there is a separate file created specially dedicated to the preprocessing tasks.</br>
### Main steps to fine-tune a transformer model
How work transformer models (encoding, picking an optmizer and loss function)?</br>
Special attention to our investigation of the best NLP models was given to the Bert models. Bert stand short for Bidirectional Encoder Representation Transformer. Inside these category are models like bert-uncased and bert-cased.
The Bert models expect encoded tensor format.
AutoModel and AutoTokenizer will take the appropriate tokeniser class and encode the input. Especially we can specify the max_length of our sentences. It is defined in the following way:
> padding="max_length", max_length=8</br>

A longer explanation of this functionality: 
When we initialise the transformer with padding = True, than the tokeniser will take the longest item(sentences) from out input and will append to the shorter instances 1's. This will take all the available data we have, for the shorter input the 1's will have no effect for the calculation of our result. The following is example for the definition of the optimizer, metrics and the loss:
> optimizer = tf.keras.optimizers.Adam(learning_rate=0.007)</br>
metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)</br>
loss = tf.keras.losses.SparseCategoricalCrossentropy()</br>

### Appropiate metrics to evaluate the results of approach.
Mostly used for evaluation was accuracy and mean squared error.
> load_accuracy = evaluate.combine(["accuracy"])</br>
  load_f1 = evaluate.combine(["f1"])

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
