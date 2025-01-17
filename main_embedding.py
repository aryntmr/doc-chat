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
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

class chat_gen():
    def __init__(self):
        self.chat_history=[]
    
    #loading document
    def load_doc(self,document_path):
        #loading the document
        loader=PyPDFLoader(document_path)
        documents=loader.load()
        #creating the splitter class object and using it to split the documents in docs
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs=text_splitter.split_documents(documents=documents)
        #creating embeddings, using vectordatabase and storing them locally, then loading from local
        embeddings=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore=FAISS.from_documents(docs,embeddings)
        vectorstore.save_local("faiss_index_datamodel")
        persisted_vectorstore=FAISS.load_local("faiss_index_datamodel",embeddings,allow_dangerous_deserialization=True)
        return persisted_vectorstore

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
            retriever=self.load_doc(r"C:\Users\arynt\OneDrive\Desktop\YeAI\research papers\A New Audio Approach Based on User Preferences Analysis to Enhance Music.pdf").as_retriever(),
            combine_docs_chain_kwargs={'prompt': prompt},
            chain_type="stuff",
        )
        # retriever=self.load_doc(r"C:\Users\arynt\OneDrive\Desktop\YeAI\research papers\A New Audio Approach Based on User Preferences Analysis to Enhance Music.pdf").as_retriever()
        # combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        # chain = create_retrieval_chain(retriever, combine_docs_chain)
        # # chain = create_history_aware_retriever(
        # #     llm, retriever, combine_docs_chain
        # # )
        return chain
    
    def ask_pdf(self,query):
        result=self.load_model()({"question":query,"chat_history":self.chat_history})
        self.chat_history.append((query,result["answer"]))
        return result['answer']
    
if __name__ == "__main__":
    chat=chat_gen()
    print(chat.ask_pdf("who is charlie chaplin"))
    print(chat.ask_pdf("when did he die?"))
