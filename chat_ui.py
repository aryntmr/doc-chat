import streamlit as st
from main_embedding import chat_gen

@st.cache_resource
def initialize():
    chat=chat_gen()
    return chat

st.session_state.chat=initialize()

st.title("Doc chat bot")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # st.chat_message("user").markdown(prompt)
    # st.session_state.messages.append({"role":"user","content":prompt})

    # response=st.session_state.chat.ask_pdf(prompt,)
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    
    # st.session_state.messages.append({"role":"user","content":prompt})

    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant's response
    response = st.session_state.chat.ask_pdf(prompt)

    # Add assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)