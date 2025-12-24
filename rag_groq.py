
import os
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/langsmith/home")
    st.session_state.docs =  st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("Groq Title")
llm = ChatGroq(groq_api_key=groq_api_key,model="qwen/qwen3-32b")

prompt = ChatPromptTemplate.from_template("""
 
                                          <context>{context}<context>
    
    Questions: {input}
""")

document_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
retriever = st.session_state.vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever=retriever,combine_docs_chain=document_chain)

val_prompt = st.text_input("Your Input here")
if val_prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input":prompt,"context":"about langchain"})
    finish_time = time.process_time() - start_time
    print("response time : " + str( finish_time))
    st.write(response['answer'])
    with st.expander("Search Similar Document"):
        for i , doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('---------------------------------------')

