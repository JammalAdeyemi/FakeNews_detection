# Fake News Detection with Machine Learning and Deep Learning
This project aims to detect `fake news` with **Machine Learning and Deep Learning**. In this project, I am using two datasets: one is from Huggingface, and the other is from Kaggle. I am concatenating these datasets to get a more comprehensive dataset for training and testing the models. I will be using the `CountVectorizer` method to convert the text data to vectors for machine learning models. Then I will be using the `Tokenizer` method to tokenize and pad the data for deep learning models. Finally, I will be comparing the accuracy of various machine learning and deep learning models to identify the best model for fake news detection.

## Folder
1. `Data`: The data used for my training was saved here.
2. `Model`: The model with the highest performance metrics was saved in this foldr.
3. `Notebook`: This folder contains the Jupyter notebooks used for training, testing, and analyzing the model.

## Dataset
I used two datasets to train and test the models.

1. `NoahGift/fake-news dataset`: This dataset contains 2,096 articles. The data is in CSV format and had multiple columns but will be making use of only this four columns: `title_without_stopwords`, `text_without_stopwords`, `type`, and `label`. I have used this dataset as one of the sources for my training and testing data.

2. `Fake and Real News Dataset`: This dataset contains two CSV files, one with fake news and the other with real news. I have used both dataset for our training and testing data by`concantenating`. This dataset has four columns: `title`, `text`, `subject`, and `date`.

## Preprocessing
I have done the following pre-processing steps before training and testing different models:

1. Concatenated the two datasets to get a more comprehensive dataset.
2. Removed the missing values from the dataset.
3. Combined the title and text columns of the dataset.
4. Converted the text data to lowercase.
5. Tokenized the text data.
6. Removed the stopwords from the text data.
7. Lemmatized the tokens.
8. Removed the punctuation marks.

## Machine Learning Models
I used both `TFIDFVectorizer` and `CountVectorizer` method to convert the text data to vecors for ML models. `CountVectorizer` had the best performance metrics compared to `TFIDFVectorizer` and I used the following machine learning models to train and test the dataset:

* Logistic Regression
* Random Forest
* Naive Bayes
* Gradient Boosting Classifier
* Decision Tree Classifier
* Extreme Gradient Boosting

After training and testing different models, I found that the `Extreme Gradient Boosting Classifier` has the highest accuracy of `98.83%`.

## Deep Learning Models
Since I am working with text data, I had to tokenize and pad the data using the `Tokenizer` and `pad_sequences` method before building the deep learning model. Afterwards, I trained and tested the dataset using the following models:

* Bi-Directional LSTM
* Bi-Directional GRU (`Gated Recurrent Units (GRUs)`)

After training and testing the model, I found that the model with the highest accuracy of 99% was `Bi-Directional LSTM`.

## Conclusion
In conclusion, I have shown that both machine learning and deep learning models can be used to detect fake news. I found that the `Extreme Gradient Boosting Classifier` has the highest accuracy among the machine learning models, while the `Bi-Directional LSTM` has the highest accuracy among the deep learning models. These models can be used in the future to detect fake news and prevent the spread of false information. This is a **Personalised Project**
