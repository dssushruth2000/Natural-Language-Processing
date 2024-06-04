## Project Overview

This combined README encompasses multiple projects, each demonstrating distinct machine learning and NLP techniques. The projects covered include Word2Vec Training and Evaluation, Named Entity Recognition with RNNs, Fine-Tuning GPT-2, and BERT Text Classification.

### Project 1: Word2Vec Training and Evaluation
- **Description**: Training Word2Vec models (CBOW and Skip-Gram) using Gensim and NLTK Brown corpus to evaluate against word similarities and Google News embeddings.
- **Setup**: Python 3.8, Libraries: nltk, gensim, numpy, matplotlib, sklearn.
- **Key Files**: `main.py`, `A1_helper.py`, `my_wv.txt`, `my_wv1.txt`, `sim.txt`, `Report.pdf`.
- **Instructions**: Run `main.py` to train models and generate embeddings; review `Report.pdf` for analysis.
- **Evaluation**: Cosine similarities comparison using Pearson correlation coefficients.

### Project 2: Named Entity Recognition with RNNs: Simple RNN vs. LSTM
- **Description**: Training and evaluating NER models using Simple RNN and LSTM to identify named entities across various datasets.
- **Setup**: Python 3.8, Libraries: tensorflow, gensim, numpy, matplotlib, sklearn.
- **Key Files**: `main.py`, `ner_dataset.csv`, `my_dataset.csv`, `sentences_and_tags.csv`, `eval_simple_rnn.txt`, `eval_lstm.txt`, `eval_lstm_custom.txt`, `Report.pdf`.
- **Instructions**: Run `main.py`; review `Report.pdf`.
- **Evaluation**: Precision, recall, and F-measure metrics.

### Project 3: Fine-Tuning GPT-2 Model
- **Description**: Fine-tuning a GPT-2 model to demonstrate dataset-specific training effects on language models.
- **Setup**: Python 3.x, pip install: transformers, datasets, tensorflow.
- **Key Files**: `new_distillgpt2.py`, `text_dataset.txt`, `report.pdf`.
- **Instructions**: Execute `new_distillgpt2.py` to run fine-tuning and text generation.

### Project 4: BERT Text Classification
- **Description**: Fine-tuning BERT model for text classification using TensorFlow and Transformers on 'emotion' dataset.
- **Setup**: Python 3.6+, pip install: tensorflow, transformers, datasets.
- **Key Files**: `A4.py`, `dataset_download.py`, `report.pdf`, `dataset_train.txt`, `dataset_test.txt`.
- **Instructions**: Execute `A4.py` to train and evaluate the model.
- **Output**: Model accuracy and examples of predictions; demonstration of BERT's contextual embeddings.

## Contributing
Contributions are welcome across all projects. Please fork the respective repository, make your changes, and submit a pull request.

Thank you for exploring these projects. For any further information or queries, please open an issue in the respective project repository.
