# Fine-Tuning GPT-2 Model

This project involves fine-tuning a GPT-2 model using the Transformers library to demonstrate the effects of dataset-specific training on language models. The fine-tuned model is compared with the original pre-trained model to illustrate differences in text generation capabilities.

## Project Overview

The project includes tasks of fine-tuning the GPT-2 model with a specific text dataset and identifying biases in the language models. The repository contains the code, dataset, and a detailed report that outlines the methodology, results, and observations.

## Installation

### Prerequisites

- Python 3.x
- pip

### Libraries

To install the required libraries, run the following command:

```bash
pip install transformers datasets tensorflow
```

## Usage

To run the fine-tuning process and generate text samples, execute the Python script `new_distillgpt2.py`. Ensure that you have the necessary data and model files in the expected directories as specified in the script.


## Repository Contents

- `new_distillgpt2.py`: The main script for training and text generation tasks.

- `text_dataset.txt`: The text dataset used for fine-tuning the model.

- `report.pdf`: Detailed report containing the methodology, examples of generated texts, and analysis of results.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

