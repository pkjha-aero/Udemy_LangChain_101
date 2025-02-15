#!/usr/bin/env python
# coding: utf-8

# # Using different models with LangChain

# In[1]:


import os


# In[11]:


os.environ['OPENAI_API_KEY'] = "..."


# In[12]:


HUGGINGFACEHUB_API_TOKEN = "..."


# In[4]:


from langchain import HuggingFaceHub


# In[5]:


from langchain_huggingface import HuggingFaceEndpoint


# In[6]:


prompt = "What are good fitness tips?"


# ## Temperature = 0.1

# In[7]:


llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text-generation", 
    temperature= 0.1, 
    max_length= 64, 
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)


# In[8]:


print(llm(prompt))


# ## Temperature = 0.9

# In[9]:


llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text-generation", 
    temperature= 0.9, 
    max_length= 64, 
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)


# In[10]:


print(llm(prompt))


# In[ ]:




