
from langchain_community.document_loaders import TextLoader

from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
import bs4

import langchain_text_splitters  as ltp

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

data_load = TextLoader("speech.txt")

data_injestion = data_load.load()
text_splitter = ltp.RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
split_data = text_splitter.split_documents(data_injestion)
#st.write(split_data)
#db = Chroma.from_documents(split_data,OllamaEmbeddings(model="llama3.2:1b"))
db = FAISS.from_documents(split_data,OllamaEmbeddings(model="llama3.2:1b"))
cresult = db.similarity_search('what attention you all need')

st.write(cresult[0].page_content)

# data_web_load = WebBaseLoader(web_path="http://lilianweng.github.io/posts/2023-06-23-agent/",                        
#                          bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title","post-content","post-header")),)      
#                          )

# webdata =  data_web_load.load()







