#!/usr/bin/env python
# coding: utf-8

# # Human as a Tool

# In[1]:


import os
#os.environ["OPENAI_API_KEY"] = "..."


# In[3]:


from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType


# ### Define LLMs

# In[4]:


llm = ChatOpenAI(temperature=0.0)
math_llm = OpenAI(temperature=0.0)


# ### Tools

# In[5]:


tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)


# ### Agent Chain

# In[6]:


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # agent="zero-shot-react-description"
    verbose=True
)


# In[7]:


agent_chain.run("What's my friend Andi's surname?")

