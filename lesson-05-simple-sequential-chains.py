#!/usr/bin/env python
# coding: utf-8

# # Simple Sequential Chains

# In[1]:


import os


# In[10]:


os.environ["OPENAI_API_KEY"] = ""


# In[3]:


from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import SimpleSequentialChain


# ### Define LLM

# In[4]:


llm = OpenAI(temperature=0)


# ### First Template

# In[5]:


first_template = "What is a good name for a company that makes {product}?"
first_prompt = PromptTemplate.from_template(first_template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)


# In[6]:


print(first_chain.run("colorful socks"))


# ### Second Template

# In[7]:


second_template = "Write a catch phrase for the following company: {company_name}"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)


# In[8]:


#print(second_chain.run("colorful socks"))
#print(second_chain.run("ABC Startup"))


# ### Overall Chain

# In[9]:


overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
catchphrase = overall_chain.run("colorful socks")
print(catchphrase)

