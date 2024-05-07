# Text Classifier and Text Clustering

This repository contains Python code for a text analysis application that demonstrates two text analysis techniques: Text Classification and Text Clustering. It uses the data in the ResearchPapers directory and calculates the TF and IDF values to make the TF-IDF vectors embeddings of the documents. This TF-IDF vector embeddings is then used for both Text Classification and Text Clustering tasks.

## Overview

- **Text Classification**: This is a supervised learning technique that assigns a label to a text document. The labels can be binary, multi-class, or multi-label. The app uses the K-Nearest Neighbors algorithm to classify the text documents.

- **Text Clustering**: This is an unsupervised learning technique that groups similar text documents together. The goal is to discover the underlying structure in the text data. The app uses the K-Means algorithm to cluster the text documents.

## Components

### 1. Weight Calculation

- **File**: `weight_calculation.py`
- **Description**: This script calculates the Term Frequency (TF), Inverse Document Frequency (IDF), and TF-IDF weights for the terms in the text documents. It preprocesses the text files, extracts stopwords, and saves the weights to CSV files.

### 2. Homepage

- **File**: `homepage.py`
- **Description**: This script sets up the homepage of the application using Streamlit. It provides an overview of the app and its functionalities.

### 3. Text Classification

- **File**: `text_classification.py`
- **Description**: This script implements the text classification functionality of the application. It trains a K-Nearest Neighbors model using TF-IDF weights and evaluates the model using accuracy, recall, F1-score, and precision metrics.

### 4. Text Clustering

- **File**: `text_clustering.py`
- **Description**: This script implements the text clustering functionality of the application. It trains a K-Means clustering model using TF-IDF weights and evaluates the model using purity, silhouette score, and adjusted Rand index metrics. It also generates an Elbow Chart and displays the top keywords for each cluster.

## Getting Started
To run the information retrieval system, follow these steps:

* Ensure you have Python 3.12 installed.
* Install NLTK and tkinter 'pip install NLTK' and 'pip install tkinter' repectively.
* Make sure Stopword-List.txt and the Research Paper directory containing all the documents is in your current working directory.
* Put the files Text_Classification and Text_Clustering in the "pages" directory
* Run the files in an IDE.
* Run this command to download the tokennizer nltk.download('punkt')
* Run the 'weights_calculation.py' script first using 'python weights_calculation.py' to create and save the weights.
* Type python -m streamlit run Homepage.py on the terminal.
* Navigate through different pages of the Streamlit GUI using the sidebar.
* Press the cross button at the top right of the app to close the app.

## Usage

* Clone the repository:
git clone https://github.com/zohaibterminator/text-analysis.git

* Install the required libraries:
pip install -r requirements.txt

* Run the streamlit app
python -m streamlit run Homepage.py

* Open the app in your browser and explore the text classification and text clustering functionalities.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

* This project was inspired by information retrieval and machine learning concepts.
* Special thanks to the developers of NLTK for providing essential natural language processing tools.
* Special thanks to developers of Streamlit that was used to develop the GUI of the app.
