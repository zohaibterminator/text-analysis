import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pages.Document_Classification import extract_weights, add_labels
pio.templates.default = 'ggplot2'


def plot_cluster(df, k):
    """

    This function is used to plot values of SSE with respect to the Number of Clusters.

    """

    if k % 2 != 0:
        k += 1

    if k < 2:
        k = 2
    
    if k+10 > 22:
        max_k = 22
    else:
        max_k = k+10

    sse = []
    iterr = range(k, max_k, 2)
    for k in iterr:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    
    df = pd.DataFrame()
    df['Number of Clusters'] = list(iterr)
    df['SSE'] = sse

    fig = px.line(df, x="Number of Clusters", y="SSE")
    st.plotly_chart(fig)


def model_train(df, n_clusters=5):
    """
    This function is used to train the model.

    Args:
        df (DataFrame): The DataFrame.
        n_clusters (int): The number of clusters. Default is 19.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++') # initialize KMeans model
    y_predict = kmeans.fit_predict(df) # fit and predict the model
    df['cluster'] = y_predict # add cluster column to the DataFrame
    return kmeans, df, y_predict # return the model, DataFrame, and predicted values


def get_top_keywords(data, clusters, labels, n_terms=10):
    """
    This function is used to get the top n keywords.

    Args:
        data (DataFrame): A Pandas DataFrame containing the TF-IDF values.
        clusters (DataFrame): A single column DataFrame containing the predicted clusters.
        labels (List): The list features of the dataframe.
        n_terms (int): The number of terms to be returned. Default is 10.
    """

    df = data.groupby(clusters).mean()

    for i, r in df.iterrows():
        st.subheader('Cluster {}'.format(i))
        st.write(', '.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


def purity_score(y_true, y_pred):
    """
    This function is used to compute the purity score.

    Args:
        y_true (List): The true labels.
        y_pred (List): The predicted labels.
    
    Returns:
        purity (float): The purity score.
    """
    contingency = contingency_matrix(y_true, y_pred) # Compute contingency matrix
    cluster_purities = np.sum(np.max(contingency, axis=0)) # Sum of the maximum values in each cluster (cluster purity)
    total_samples = np.sum(contingency) # Total number of samples
    purity = cluster_purities / total_samples # Compute purity score
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
    metrics["Purity"] = purity_score(df['label'], y_predict) # compute purity score
    metrics["Silhouette Score"] = silhouette_score(df, y_predict) # evaluate clustering using silhouette score
    metrics["Random Index"] = adjusted_rand_score(df['label'], y_predict) # compute random index
    return metrics


def main():
    """
    This function is the main function that runs the app.

    """
    if "n_clusters" not in st.session_state: # check if the number of clusters is in the session state
        st.session_state["n_clusters"] = 5 # if not, set the default number of clusters to 5
    if "iterr" not in st.session_state: # check if the range is in the session state
        st.session_state["iterr"] = range(2, 12, 2) # if not, set the default range to 2, 12, 2
    if "sse" not in st.session_state: # check if the SSE is in the session state
        st.session_state["sse"] = [0.0, 0.0, 0.0, 0.0, 0.0] # if not, set the default SSE to 0.0, 0.0, 0.0, 0.0, 0.0
    if "top_k" not in st.session_state: # check if the top k is in the session state
        st.session_state["top_k"] = 10 # if not, set the default top k to 10

    n_clusters = st.text_input("Enter the number of clusters you want to set:", st.session_state["n_clusters"]) # get the number of clusters
    submit = st.button("Submit") # create a submit button

    df = extract_weights() # extract the TF-IDF weights
    model, df, y_predict = model_train(df, int(n_clusters)) # train the model based on the number of clusters given by the user
    metrics = clust_metrics(df, df['cluster']) # compute the evaluation metrics

    st.header("Metrics:") # create a header for the Metrics
    for key, value in metrics.items(): # loop through the metrics
        with st.container(): # create a container containing the metrics and their values
            st.subheader(f"{key}:") 
            st.write(f"{value}")

    st.header("Elbow Chart:") # create a header for the Elbow Chart
    plot_cluster(df, int(n_clusters)) # plot the Elbow Chart

    top_k = st.text_input("Enter the number of top keywords you want to see from each cluster:", st.session_state["top_k"]) # get the number of top keywords
    submit = st.button("Submit the number") # create a submit button

    st.header("Top Keywords:") # create a header for the Top Keywords
    get_top_keywords(df, y_predict, extract_weights().columns, int(top_k)) # get the top keywords for each cluster based on the number of top keywords given by the user


if __name__ == "__main__":
    main()