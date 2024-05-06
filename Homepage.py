import streamlit as st


st.set_page_config(
    page_title="Home",
    page_icon="📚",
)

st.title("Text-Classifier and Text-Clustering")

st.subheader("Welcome to the App!")
st.markdown("This app is designed to demonstrate two text analysis techniques: Text Classification and Text Clustering")
st.markdown("- Text Classification: This is a supervised learning technique that assigns a label to a text document. The labels can be binary, multi-class, or multi-label. The goal is to predict the label of a text document based on its content. The app uses the K-Nearest Neighbors algorithm to classify the text documents.")
st.markdown("- Text Clustering: This is an unsupervised learning technique that groups similar text documents together. The goal is to discover the underlying structure in the text data. The app uses the K-Means algorithm to cluster the text documents.")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    padding-left:40px;
}
</style>
''', unsafe_allow_html=True)

st.markdown("To get started, select one of the options from the sidebar")