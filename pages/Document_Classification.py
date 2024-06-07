import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


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

    labels = []

    for index in tf_idf.index:
        if index < 8:
            labels.append(1)
        elif index < 12:
            labels.append(2)
        elif index < 17:
            labels.append(3)
        elif index < 22:
            labels.append(4)
        else:
            labels.append(5)

    tf_idf['true_label'] = labels
    tf_idf.to_csv('labelled.csv') # save the DataFrame to a CSV file

    return tf_idf


def model_train(tf_idf, n_neighbors=6):
    """
    This function is used to train the model.

    Args:
        tf_idf (DataFrame): The TF-IDF DataFrame.

    Returns:
        None
    """

    # split the data into training and testing sets
    y = tf_idf['true_label'].apply(int) # convert the labels to integers
    X = tf_idf.drop('true_label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42) # 70% training and 30% testing

    # train the model
    model = KNeighborsClassifier(n_neighbors=int(n_neighbors)) # create a KNN model

    start_time = time.time() # start the timer
    model.fit(X_train, y_train) # train the model
    end_time = time.time() # end the timer

    return model, X_test, y_test, end_time - start_time # return the trained model, the testing data, and the testing labels


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

    start_time = time.time() # start the timer
    y_pred = model.predict(X_test) # test the model
    end_time = time.time() # end the timer

    metric = {} # initialize a dictionary to store the metrics

    # evaluate the model
    metric['Accuracy'] = accuracy_score(y_test, y_pred) # compute accuracy
    metric['Recall'] = recall_score(y_test, y_pred, average='macro') # compute recall
    metric['F1-score'] = f1_score(y_test, y_pred, average='macro') # compute F1-score
    metric['Precision'] = precision_score(y_test, y_pred, average='macro') # compute precision

    return metric, end_time-start_time # return the metrics


def main():
    """
    This function is the main function that runs the app.

    """

    if "n_neighbors" not in st.session_state: # check if the number of neighbors is in the session state
        st.session_state["n_neighbors"] = 6 # if not, set the default number of neighbors to 6

    n_neighbors = st.text_input("Enter the number of neighbors you want to set:", st.session_state["n_neighbors"]) # get the number of neighbors
    submit = st.button("Submit") # create a submit button

    df = extract_weights() # extract the TF-IDF weights
    df = add_labels(df) # add the labels to the DataFrame

    model, X_test, y_test, model_training_time = model_train(df, n_neighbors) # train the model based on the number of neighbors given by the user
    metrics, model_eval_time = eval_metric(model, X_test, y_test) # evaluate the model

    for key, value in metrics.items(): # loop through the metrics
        with st.container(): # create a container containing the metrics and their values
            st.subheader(f"{key}:")
            st.write(f"{round(value, 3) * 100}%")

    st.subheader("Model Time:")
    st.write(f"{round(model_training_time + model_eval_time, 3)} sec") # display the model training time


if __name__ == "__main__":
    main()