#!/usr/bin/env python
# coding: utf-8

# # Prompts and LLMs

# In[1]:


import os


# In[12]:


os.environ["OPENAI_API_KEY"] = "..."


# In[3]:


from langchain.llms import OpenAI


# ## Deterministic Response

# In[4]:


llm = OpenAI(temperature=0) # Default model is text-davinci-003


# In[5]:


prompt = "What would a good company name be for a company that makes colorful socks?"


# In[6]:


print(llm(prompt))


# In[7]:


result = llm.generate([prompt]*5)


# In[8]:


result


# In[9]:


for company_name in result.generations:
    print(company_name[0].text)


# ## Probabilistic Response

# In[10]:


llm = OpenAI(temperature=0.9)
result = llm.generate([prompt]*5)
for company_name in result.generations:
    print(company_name[0].text)

