import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, rand_score
from pages.Document_Classification import extract_weights, add_labels
pio.templates.default = 'ggplot2'


def plot_cluster(df, k):
    """
    This function is used to plot values of SSE with respect to the Number of Clusters.

    """

    if k < 2:
        k = 2 # if k is less than 2, set k to 2

    if k+10 > 21: # if k+10 is greater than 21, set max_k to 21 (max 21 clusters are allowed)
        max_k = 21
    else: # otherwise, set max_k to k+10 (for elbow chart)
        max_k = k+10

    sse = [] # initialize an empty list to store the SSE values
    iterr = range(k, max_k, 1) # create a range of values from k to max_k
    for k in iterr:
        kmeans = KMeans(n_clusters=k, random_state=42) # initialize KMeans model
        kmeans.fit(df) # fit the model
        sse.append(kmeans.inertia_) # append the SSE value to the list
    
    df = pd.DataFrame() # create a DataFrame
    df['Number of Clusters'] = list(iterr) # add the Number of Clusters to the DataFrame
    df['SSE'] = sse # add the SSE values to the DataFrame

    fig = px.line(df, x="Number of Clusters", y="SSE") # create a line plot
    st.plotly_chart(fig) # plot the line plot


def model_train(df, n_clusters=9):
    """
    This function is used to train the model.

    Args:
        df (DataFrame): The DataFrame.
        n_clusters (int): The number of clusters. Default is 8.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++') # initialize KMeans model

    start_time = time.time() # start the timer
    y_predict = kmeans.fit_predict(df) # fit and predict the model
    end_time = time.time() # end the timer

    df['cluster'] = y_predict # add cluster column to the DataFrame

    return kmeans, df, y_predict, end_time-start_time # return the model, DataFrame, predicted values, and model training time


def purity_score(y_true, y_pred):
    """
    This function is used to compute the purity score.

    Args:
        y_true (List): The true labels.
        y_pred (List): The predicted labels.

    Returns:
        purity (float): The purity score.
    """

    contingency = contingency_matrix(y_true, y_pred) # compute contingency matrix
    cluster_purities = np.sum(np.max(contingency, axis=0)) # sum of the maximum values in each cluster (cluster purity)
    total_samples = np.sum(contingency) # total number of samples
    purity = cluster_purities / total_samples # compute purity score
    return purity


def clust_metrics(df, y_predict):
    """
    This function is used to compute the evaluation metrics.

    Args:
        df (DataFrame): A Pandas DataFrame representing the TF-IDF values.
        y_predict (List): The predicted labels.

    Returns:
        metrics (Dict): A dictionary containing the evaluation metrics.
    """

    metrics = {} # initialize a dictionary to store the metrics

    df = add_labels(df) # add the true labels to the DataFrame

    # print evaluation metrics
    metrics["Purity"] = purity_score(df['true_label'], y_predict) # compute purity score
    metrics["Silhouette Score"] = silhouette_score(df, y_predict) # evaluate clustering using silhouette score
    metrics["Random Index"] = rand_score(df['true_label'], y_predict) # compute random index
    return metrics


def main():
    """
    This function is the main function that runs the app.

    """

    if "n_clusters" not in st.session_state: # check if the number of clusters is in the session state
        st.session_state["n_clusters"] = 9 # if not, set the default number of clusters to 8
    if "iterr" not in st.session_state: # check if the range is in the session state
        st.session_state["iterr"] = range(2, 12, 1) # if not, set the default range to 2, 12, 2
    if "sse" not in st.session_state: # check if the SSE is in the session state
        st.session_state["sse"] = [0.0, 0.0, 0.0, 0.0, 0.0] # if not, set the default SSE to 0.0, 0.0, 0.0, 0.0, 0.0
    if "top_k" not in st.session_state: # check if the top k is in the session state
        st.session_state["top_k"] = 10 # if not, set the default top k to 10

    n_clusters = st.text_input("Enter the number of clusters you want to set:", st.session_state["n_clusters"]) # get the number of clusters
    submit = st.button("Submit") # create a submit button

    df = extract_weights() # extract the TF-IDF weights
    model, df, y_predict, model_time = model_train(df, int(n_clusters)) # train the model based on the number of clusters given by the user
    metrics = clust_metrics(df, df['cluster']) # compute the evaluation metrics

    st.header("Metrics:") # create a header for the Metrics
    for key, value in metrics.items(): # loop through the metrics
        with st.container(): # create a container containing the metrics and their values
            st.subheader(f"{key}:")
            st.write(f"{round(value, 3) * 100}%")

    st.subheader("Model Time:") # create a subheader for the Model Training Time
    st.write(f"{round(model_time, 3)} sec") # display the model training time

    st.header("Elbow Chart:") # create a header for the Elbow Chart
    plot_cluster(df, int(n_clusters)) # plot the Elbow Chart


if __name__ == "__main__":
    main()