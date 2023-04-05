# Naive-Bayes-Classifier

## Introduction 
This project implemented Multinomial Naive Bayes and used the model to filter (classify) spam messages on a given dataset.

## General info
- Language: Python 
- Dataset: [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/)

## Repository's structure
- `main.py`: implementation of the algorithm. 
- `spam_filtering.ipynb`: unit test and usage of the model on the given dataset.
- `utils.py`: helper functions and classes for the algorithm.
- `requirements.txt`: essential packages of the project.

## Details
1. Implementation of the algorithm: 
    - Assumed equal prior probabilities of spam and non-spam emails. 
    - Computed sum of log probabilities to avoid *underflow*.
    - Used Lidstone smoothing as pseudocount to prevent zero probabilities.
3. Testing and using the model: 
    - Unit tested the model.
    - Applied the model on the [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) dataset.
