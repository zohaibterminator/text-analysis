import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


st.set_page_config(
    page_title="Document Classification",
    page_icon="📚",
)


def extract_weights():
    """
    This function is used to extract the TF-IDF and the IDF weights from their respective csv files.

    Returns:
        TF-IDF (DataFrame): The extracted TF-IDF weights.
        IDF (DataFrame): The extracted IDF weights.
    """

    TF_IDF = pd.read_csv('tf-idf.csv', index_col=0)  # read TF-IDF DataFrame from CSV

    return TF_IDF


def add_labels(tf_idf):
    """
    This function is used to add the labels to the TF-IDF DataFrame.

    Args:
        tf_idf (DataFrame): The TF-IDF DataFrame.

    Returns:
        tf_idf (DataFrame): The TF-IDF DataFrame with labels.
    """

    label1 = ['1', '2', '3', '7']
    label2 = ['8', '9', '11']
    label3 = ['12', '13', '14', '15', '16']
    label4 = ['17', '18', '21']
    label5 = ['22', '23', '24', '25', '26']

    for index in tf_idf.index:
        if index in label1:
            tf_idf['label'] = 1
        elif index in label2:
            tf_idf['label'] = 2
        elif index in label3:
            tf_idf['label'] = 3
        elif index in label4:
            tf_idf['label'] = 4
        elif index in label5:
            tf_idf['label'] = 5

    return tf_idf


def model_train(tf_idf, n_neighbors=5):
    """
    This function is used to train the model.

    Args:
        tf_idf (DataFrame): The TF-IDF DataFrame.

    Returns:
        None
    """

    # split the data into training and testing sets
    y = tf_idf['label'].apply(int) # convert the labels to integers
    X = tf_idf.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% testing

    # train the model
    model = KNeighborsClassifier(n_neighbors=int(n_neighbors)) # create a KNN model
    model.fit(X_train, y_train) # train the model

    return model, X_test, y_test


def eval_metric(model, X_test, y_test):
    """
    This function is used to evaluate the model.

    Args:
        model (Model): The trained model.
        X_test (DataFrame): The testing data.
        y_test (DataFrame): The testing labels.
    
    Returns:
        metric (Dict): A dictionary containing the evaluation metrics.
    """

    y_pred = model.predict(X_test) # test the model

    metric = {} # initialize a dictionary to store the metrics

    # evaluate the model
    metric['Accuracy'] = accuracy_score(y_test, y_pred) # compute accuracy
    metric['Recall'] = recall_score(y_test, y_pred, average='weighted') # compute recall
    metric['F1-score'] = f1_score(y_test, y_pred, average='weighted') # compute F1-score
    metric['Precision'] = precision_score(y_test, y_pred, average='weighted') # compute precision

    return metric # return the metrics


def main():
    """
    This function is the main function that runs the app.

    """

    if "n_neighbors" not in st.session_state: # check if the number of neighbors is in the session state
        st.session_state["n_neighbors"] = 5 # if not, set the default number of neighbors to 5

    n_neighbors = st.text_input("Enter the number of neighbors you want to set:", st.session_state["n_neighbors"]) # get the number of neighbors
    submit = st.button("Submit") # create a submit button

    df = extract_weights() # extract the TF-IDF weights
    df = add_labels(df) # add the labels to the DataFrame

    model, X_test, y_test = model_train(df, n_neighbors) # train the model based on the number of neighbors given by the user
    metrics = eval_metric(model, X_test, y_test) # evaluate the model

    for key, value in metrics.items(): # loop through the metrics
        with st.container(): # create a container containing the metrics and their values
            st.subheader(f"{key}:")
            st.write(f"{value}")


if __name__ == "__main__":
    main()