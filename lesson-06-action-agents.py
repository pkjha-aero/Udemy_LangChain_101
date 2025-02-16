#!/usr/bin/env python
# coding: utf-8

# # Action Agents

# In[1]:


import os


# In[2]:


#os.environ["OPENAI_API_KEY"] = "..."


# In[4]:


import pprint
from langchain.llms import OpenAI
from langchain.agents import load_tools, get_all_tool_names, initialize_agent
#from langchain_community.agent_toolkits.load_tools import get_all_tool_names


# In[5]:


prompt = "When was the 3rd president of the United States born? What is that year raised to the power of 3?"


# In[6]:


pp = pprint.PrettyPrinter(indent = 4)
pp.pprint(get_all_tool_names())


# ### Define LLM

# In[7]:


llm = OpenAI(temperature=0)


# ### Define Tools and Agent

# In[8]:


tools = load_tools(["wikipedia", "llm-math"], llm=llm)


# In[9]:


agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# In[10]:


agent.run(prompt)

