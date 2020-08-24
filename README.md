

  <h1 align="center">Sentiment Analysis for Ignition Hacks 2020</h1>

  <p align="center">
    Rigorous Preprocessing and Sentence Encoding NLP Algorithm
    <br />
    <a href="https://github.com/Positron23/ignitionhacks">
    <strong>Documentation</strong></a>
    <br />
    
[![Contributors][contributors-shield]](https://github.com/Positron23/ignitionhacks/graphs/contributors)
[![MIT License][license-shield]](https://github.com/Positron23/ignitionhacks/blob/master/LICENSE.txt)
## Table of Contents

* [Problem Statement](#problem-statement)
	 * [Built With](#built-with)
* [Introduction](#introduction)
	 * [Algorithm Design](#algorithm-design)
 * [Preprocessing](#preprocessing)
* [Encoding](#encoding)
* [Next Steps](#next-steps)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Problem Statement

The prompt for the 2020 Ignition Hacks Sigma division was to perform sentiment analysis on a random selection of tweets, with the goal of creating a machine learning model to find if tweets carried positive or negative sentiment. This is a binary classification problem involving natural language processing, with 1,000,000 labeled data points available for training and validation, and 600,000 unlabeled data points which are used to judge models.

### Built With
* Python
* Tensorflow
* Numpy
* Pandas
* NLTK

## Introduction
The first part of creating a model to perform natural language processing was to decide on the tools. By the rules of the competition, Python was required to be the only allowed programming language. Due to the large amount of online resources, as well as past experience from group members, it was decided that Tensorflow was to be used to construct the model, utilizing additional Python data science modules such as pandas and numpy. The scikit-learn model was not used due to the smaller amount of experimentation and customization which could be done in terms of the model layers. In addition, while pretrained models such as RoBERTa would likely provide superior results to raw Tensorflow code, the training time would be beyond the limits of the hackathon and the available resources, and freezing the layers would go against the rule of not being allowed to use data from outside the hackathon.

The large amount of training data, as well as the lack of sentiment clarity in many tweets (as manually observed), led us to use deep learning models, with a larger number of deep learning layers. This would allow the model to learn more subtle patterns within the data, and make full use of the dataset.
Due to there only being one feature, the tweet itself, it was difficult to collect information about the dataset without using natural language processing tools. It was, however, noted that in the training data, there was an exact 50-50 split between tweets which were labeled to be of positive sentiment and tweets of negative sentiment. Therefore, it was decided that the split between training and validation data would be done at random, with approximately 15% of the data set to be validation data and the remaining 85% to be training data.
### Algorithm Design
![Algorithm Design](https://raw.githubusercontent.com/Positron23/ignitionhacks/master/ImageAssets/Dissertation.png)
## Preprocessing
Due to the nature of tweets and other fast text message based communication, it was imperative that preprocessing was necessary for effective natural language processing (NLP) and deep learning.

The most fundamental and perhaps most effective approach is lowercasing all the text data. Although this technique is more useful with sparse instances of words in smaller datasets, lowercasing still proved beneficial by improving validation accuracy by [REPLACE]%.

Tweets often contain expressions that may not contribute to the overall sentiment, such as user handles (@janedoe), hashtags (#ignitionhacks2020), and links ([www.ignitionhacks.org](http://www.ignitionhacks.org)). These expressions often contribute to greater noise in the dataset since many require additional context, such as understanding a user’s history or reading what is on the linked webpage, to provide a substantive sentiment relation. It is important to note that, in this dataset, it was found that hashtags were beneficial for understanding sentiment, at least by an empirical measure of the accuracy metric. A possible hypothesis is that certain emotions were associated with these tags, and could in fact be used for sentiment analysis on its own. Regex was used to clean up the data by removing handles and links, as well as punctuation and numbers.

To further reduce noise and improve sentiment analysis performance, stopword removal, normalization, and stemming was used. The English language contains many short and common words that do not give additional context for NLP, such as ‘a’ or ‘the’. These stopwords were removed by tokenizing the sentences and checking for matches against a list provided by NLTK. To keep normalization times low, a simple NFKD (Normalization Form Compatibility Decomposition) unicode normalization was implemented to remove special characters. Afterwards, the Porter stemming algorithm reduced words to their root form, allowing for more consistent sentence encoding.

## Encoding
 Most NLP algorithms utilize a basic tokenization and a pretrained embedding layer such as ‘keras.embedding’. In order to improve upon this and create a model that could be effective on different types of text data, word vectorization and sentence encoding was tested. Both methods convert strings of text into vectors of floats based on semantic similarity, given by techniques such as cosine or euclidean distance. This is superior to conventional embedding layers as it gives a measure of similarity between different words in the case of word embedding, as well as word-sentence context in the case of sentence embedding.

A disadvantage of using word embedding is losing context and thus being susceptible to spelling mistakes, expressions that require multiple words, abbreviations, and words with similar meanings. After testing, sentence encoding proved to be a better approach, and consistently provided close to a 2% increase in validation accuracy compared to the same model with default keras embedding. However, it must be noted that these encodings came at the cost of long run times and there are also averaging techniques to create contextual relationships between word vectors. Nonetheless, it is important to focus on the real strength of sentence encoding, which is providing associations between words such as “Ignition Hacks” and “Hackathon”, as well as greater flexibility for the model to handle new data or train on other languages.

## Modelling

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

## Next Steps



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Thank you to Ignition Hacks 2020 for a wonderful challenge and experience!
* Tensorflow Hub for Universal Sentence Encoder
* KDnuggets for NLP preprocessing background information: https://www.kdnuggets.com/2019/04/text-preprocessing-nlp-machine-learning.html

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
