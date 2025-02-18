#!/usr/bin/env python
# coding: utf-8

# # Document Loading and QA Retrieval

# ### Imports

# In[1]:


import os
#os.environ["OPENAI_API_KEY"] = "..."


# In[3]:


from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma # Vector DB
from langchain.chains import RetrievalQA


# ### Loader and Documents

# In[4]:


loader = TextLoader("./state-of-the-union-23.txt")
documents = loader.load()


# ### Text Splitter

# In[5]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)


# ### Embeddings and Store

# In[6]:


embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")


# ### LLM and Chain

# In[7]:


llm = OpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())


# In[8]:


print(chain.run("What did biden talk about Ohio?"))


# In[9]:


print(chain.run("What did biden talk about China?"))

