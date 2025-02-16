#!/usr/bin/env python
# coding: utf-8

# # Prompt Templating and Chaining

# ## Reproducible way to generate a prompt

# In[1]:


import os


# In[16]:


os.environ["OPENAI_API_KEY"] = "..."


# In[3]:


from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


# ### Define LLM

# In[4]:


llm = OpenAI(temperature=0.9)


# ### Template with one variable

# In[5]:


template = "You are a naming consultant for new companies. What is a good name for a company that makes {product}?"


# In[6]:


prompt = PromptTemplate.from_template(template)


# In[7]:


print(prompt.format(product = "colorful socks"))


# In[8]:


chain = LLMChain(llm=llm, prompt=prompt)


# In[9]:


print(chain.run({"product": "colorful socks"}))


# In[10]:


# This achieves the same thing as previous cell. Passing the string instead of dictionary
print(chain.run("colorful socks"))


# ### Template with multiple variables

# In[11]:


template = "You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?"


# In[12]:


prompt = PromptTemplate.from_template(template)


# In[13]:


print(prompt.format(company = "Startup", product = "colorful socks"))


# In[14]:


chain = LLMChain(llm=llm, prompt=prompt)


# In[15]:


print(chain.run({"company": "ABC Startup", "product": "colorful socks"}))

