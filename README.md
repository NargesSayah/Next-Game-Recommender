# Next-Game Recommender System
Predicts the next game label based on the user's history

This README provides an overview of a Game Recommendation System designed to predict game labels or recommend next games based on game descriptions. The system utilizes deep learning models, specifically BERT (Bidirectional Encoder Representations from Transformers), for processing and classifying game descriptions into categories that can help in recommending similar games or predicting the genre of a new game. It leverages GPU acceleration using CUDA for efficient training and inference.

## Installation

Before running the code, ensure that the following packages are installed:

```bash
pip install transformers
pip install beautifulsoup4
pip install lxml
```

## System Overview

The system operates by processing game descriptions, extracting features using a pre-trained BERT model, and classifying them into different game categories. It is designed to run on Google Colab and requires mounting a Google Drive for accessing the dataset and saving models. It exploits GPU acceleration via CUDA for enhanced performance during model training and evaluation.

## Data Preparation

The dataset consists of game descriptions in Farsi (description_fa) along with their corresponding labels. The data is split into training and testing sets, located at specified paths in Google Drive. Data preprocessing involves cleaning HTML tags from descriptions using BeautifulSoup and converting texts into a format suitable for BERT processing.

## Model

The system employs a custom neural network model that integrates BERT for feature extraction and a series of linear layers for classification. It is capable of classifying game descriptions into multiple categories, relying on the BERT model HooshvareLab/bert-fa-base-uncased for understanding the context of Farsi descriptions. The model utilizes CUDA for running computations on a GPU, significantly speeding up the training and inference processes.

## Training and Evaluation

Training involves fine-tuning the BERT model on the game description dataset with specified hyperparameters, including learning rate, epochs, and batch size. The system uses the AdamW optimizer with a linear scheduler for learning rate adjustment and exploits GPU acceleration for efficient model training. It evaluates model performance using accuracy, confusion matrix, and classification report metrics, ensuring the model effectively classifies the game descriptions.

## File Structure

- `TRAIN_PATH`, `TEST_PATH`: Paths to the training and testing dataset CSV files in Google Drive.
- `SAVE_PATH`: Path to save the trained models.
- `LOAD_PATH`: Path to load a pre-trained model for continuing training (optional).

## Usage
To run the system, ensure the dataset is correctly placed in Google Drive and paths in the script are correctly set. The system is executed as a Python script, automatically training and evaluating the model based on the provided dataset. It requires a CUDA-compatible GPU for efficient execution.

## Saving and Loading Models
The system saves models at specified epochs and the final model after training completion to the Google Drive. These models can be loaded for further training or inference, leveraging CUDA for efficient operations.

## Conclusion
This Game Recommendation System leverages the power of BERT and deep learning to classify and recommend games based on their descriptions, utilizing GPU acceleration via CUDA for enhanced performance. It is designed to be flexible, allowing for adjustments to model architecture and training parameters as needed, making it a robust solution for game recommendation and classification tasks.
