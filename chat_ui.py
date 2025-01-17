import streamlit as st
from main_embedding import chat_gen

@st.cache_resource
def initialize():
    return chat_gen()

st.session_state.chat = initialize()

st.title("Doc Chat Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader for user documents
uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])
if uploaded_file:
    st.session_state.chat.load_doc(uploaded_file)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document:"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant's response
    if uploaded_file:
        response = st.session_state.chat.ask_pdf(prompt)
    else:
        response = "Please upload a document first."

    # Add assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
