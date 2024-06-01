# Named Entity Recognition with RNNs: Simple RNN vs. LSTM

This repository contains the code and resources for training and evaluating Named Entity Recognition (NER) models using Simple RNN and LSTM (Long Short-Term Memory) layers. These models are assessed on their ability to accurately identify and classify named entities in different datasets.

## Project Description

This project explores the effectiveness of two RNN architectures—Simple RNN and LSTM—in performing Named Entity Recognition. The models are trained and tested using the `ner_dataset.csv` and `my_dataset.csv` files, focusing on a variety of entity types such as geographical locations, persons, organizations, and time expressions. The primary goal is to compare these models based on precision, recall, and F-measure across different entity types and datasets.

## Setup

**Python Version:** 3.8

**Libraries Required:**
- tensorflow
- gensim
- numpy
- matplotlib (for potential plotting)
- sklearn (if used for preprocessing or evaluation metrics)

## Repository Structure

- `main.py`: Main script for training the models and conducting evaluations.
- `ner_dataset.csv`: Dataset file containing training and testing data for NER.
- `my_dataset.csv`: Custom dataset file for additional model testing.
- `sentences_and_tags.csv`: Manually created dataset for evaluating model performance on custom sentences.
- `eval_simple_rnn.txt`, `eval_lstm.txt`, `eval_lstm_custom.txt`: Evaluation results for the Simple RNN, LSTM, and customized LSTM models.
- `Report.pdf`: Detailed report that describes the methodologies, model comparisons, and key findings.

## Instructions

To replicate the project results, follow these steps:

1. Ensure Python 3.8 and all required libraries are installed.
2. Run the `main.py` script to train the models and perform evaluations: `python main.py`.
3. Review the `Report.pdf` for an in-depth analysis of model performance and insights derived from the evaluations.

## Visualization

This project may include scripts for visualizing the training process, model performance, or other relevant data visualizations to better understand the results.

## Evaluation

Model performances are evaluated using precision, recall, and F-measure metrics, which are detailed in the corresponding evaluation files. These metrics help illustrate the effectiveness of each model in recognizing and classifying named entities accurately.

## Contributing

Contributions are welcome! If you have improvements or additional features, please fork the repository, make your changes, and submit a pull request.
