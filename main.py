# Streamlit app
st.title("Topic Modeling App")
text = st.text_input("Enter some text here:")
if text:
    # Create a list of documents
    documents = [text, text]  # Duplicate the input text

    # Fit the BERTopic model on the documents
    topics, _ = topic_model.fit_transform(documents)

    # Display the predicted topic for the first document
    st.write(f"Predicted topic: {topics[0]}")
