# Sentiment Analysis on IMDB Reviews

## Project Overview

This project aims to build a sentiment analysis model to classify movie reviews into **positive** or **negative** categories. Using Natural Language Processing (NLP) techniques, the project processes and analyzes text data to predict the sentiment of the reviews. The model uses **TF-IDF** for feature extraction and various machine learning classifiers such as **Logistic Regression** and **Naive Bayes** for sentiment classification. 

## Dataset

The dataset used in this project consists of movie reviews collected from IMDB. It is split into three parts:

- **Train.csv**: The training dataset used to train the model.
- **Test.csv**: The test dataset used to evaluate the model's performance.
- **Valid.csv**: The validation dataset used to fine-tune and validate the model.

Each dataset contains two columns: 
- **text**: The movie review text.
- **label**: The sentiment label, where 1 represents a positive review and 0 represents a negative review.

## Installation

To run this project locally, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-imdb.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sentiment-analysis-imdb
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have the necessary NLTK data files:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage

### Data Loading

The dataset can be loaded using **Pandas** by reading the CSV files into DataFrames.

### Data Preprocessing

Before training the model, the review text is cleaned by:
- Removing HTML tags and unnecessary characters.
- Tokenizing the text into individual words.
- Removing stop words (common words such as "the", "is", etc., which do not contribute much to sentiment).
- Lemmatizing words (converting them to their base forms, e.g., "running" to "run").

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned text into numerical features that can be fed into the machine learning models.

### Model Training

The sentiment analysis model is trained using machine learning classifiers, such as **Logistic Regression** or **Naive Bayes**, on the training dataset.

### Model Evaluation

The model's performance is evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. The results are visualized with:
- **Confusion Matrix**: Displays the number of true positives, false positives, true negatives, and false negatives.
- **ROC Curve**: Visualizes the performance of the model by plotting the true positive rate against the false positive rate.

## Visualizations

Several visualizations are generated to understand the dataset and model performance:
- **Word Cloud**: Displays the most frequent words in the dataset before and after preprocessing.
- **Class Distribution**: Visualizes the distribution of positive and negative reviews in the training dataset.
- **Confusion Matrix**: Helps evaluate how well the model is performing by comparing predicted vs. actual labels.
- **ROC Curve**: Displays the trade-off between true positive rate and false positive rate.

## Model Performance

After training the model, the following metrics were achieved:

- **Accuracy**: 89%
- **Precision**: 90%
- **Recall**: 85%
- **F1-score**: 87%

## Contributions

Feel free to contribute to this project by forking the repository and submitting pull requests. Contributions are highly appreciated!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
