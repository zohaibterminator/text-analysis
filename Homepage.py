import streamlit as st


st.set_page_config(
    page_title="Home",
    page_icon="📚",
)

st.title("Document-Classifier and Document-Clustering")

st.subheader("Welcome to the App!")
st.markdown("This app is designed to demonstrate two text analysis techniques: Document Classification and Document Clustering")
st.markdown("- Document Classification: Document classification involves categorizing documents into predefined classes or categories based on their content. This task is often supervised, meaning that the classification model is trained on labeled data, where each document is associated with its correct class or category. The app uses the K-Nearest Neighbors algorithm to classify the text documents.")
st.markdown("- Document Clustering: Document clustering, on the other hand, is an unsupervised task where documents are grouped into clusters based on their similarity. Clustering algorithms identify natural groupings within the documents without any prior knowledge of class labels. The app uses the K-Means algorithm to cluster the text documents.")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

st.markdown("To get started, select one of the options from the sidebar")