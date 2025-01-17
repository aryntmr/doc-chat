import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import retrieval_qa
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
import streamlit as st

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] or st.secrets.get("OPENAI_API_KEY")
import tempfile

class chat_gen():
    def __init__(self):
        self.chat_history=[]
    
    # loading document
    def load_doc(self,uploaded_file):
        # Check if uploaded_file is a Streamlit UploadedFile object
        if hasattr(uploaded_file, "read"):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
        else:
            # Assume uploaded_file is a file path
            temp_file_path = uploaded_file

        # loading the document
        loader=PyPDFLoader(temp_file_path)
        documents=loader.load()

        # creating the splitter class object and using it to split the documents in docs
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs=text_splitter.split_documents(documents=documents)

        # creating embeddings, using vectordatabase and storing them locally, then loading from local
        embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.vectorstore = FAISS.from_documents(docs, embeddings)

    def load_model(self,):
        llm=ChatOpenAI(
            temperature=0,
            max_tokens=4000
        )

        system_instruction="""
            As an AI assistant, you must answer the query from the user from the retrieved content, if no relavant information is available, answer the question by using your knowledge about the topic
            """
        
        template=(
            f"{system_instruction}"
            "Combine the chat history{chat_history} and follow up question into "
            "a standalone question to answer from the {context}"
            "Follow up question:{question}"
        )
        prompt=PromptTemplate.from_template(template)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            combine_docs_chain_kwargs={'prompt': prompt},
            chain_type="stuff",
        )

        return chain
    
    def ask_pdf(self,query):
        result=self.load_model()({"question":query,"chat_history":self.chat_history})
        self.chat_history.append((query,result["answer"]))
        return result['answer']
    
if __name__ == "__main__":
    chat=chat_gen()
    print(chat.ask_pdf("who is charlie chaplin"))
    print(chat.ask_pdf("when did he die?"))
