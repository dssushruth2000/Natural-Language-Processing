# BERT Text Classification

This project demonstrates the fine-tuning of a BERT model for text classification using TensorFlow and the Transformers library. The model is trained and evaluated on the 'emotion' dataset from the Hugging Face hub, which consists of text data labeled with different emotional content.

## Installation

### Prerequisites
- Python 3.6+
- pip

### Libraries
Install the necessary Python libraries using pip:

```bash
pip install tensorflow
pip install transformers
pip install datasets
```

## Dataset
The dataset used is 'dair-ai/emotion', which can be directly loaded using the Hugging Face datasets library. The dataset is split into training and testing sets, each containing over 1000 examples.

## Model
The model used is 'distilbert-base-uncased' from Hugging Face's Transformers library. This BERT model is initially non-trainable and fine-tuned on the emotion dataset.

## Usage
To run the training and evaluation script, execute the following command:

```bash
python A4.py
```

## Files
- `A4.py`: Contains the code to load the dataset, fine-tune the BERT model, and evaluate its performance.
- `report.pdf`: Includes a detailed report of the project outcomes, methodology, and observations regarding the model's performance and contextual embeddings.
- `dataset_train.txt`: Text file containing the training data.
- `dataset_test.txt`: Text file containing the testing data.

## Output
The script will output the accuracy of the model on the test dataset and examples of correct and incorrect predictions. It also includes demonstrations of BERT's contextual embeddings with cosine similarity scores.
