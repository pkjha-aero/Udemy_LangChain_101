#!/usr/bin/env python
# coding: utf-8

# # Plan and Execute Agents

# ### Imports

# In[1]:


import os
#os.environ['SERPAPI_API_KEY'] = "..."
#os.environ["OPENAI_API_KEY"] = "..."


# In[3]:


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain import SerpAPIWrapper, LLMMathChain, WikipediaAPIWrapper
from langchain_core.tools import Tool


# In[4]:


prompt = "Where will the summer olympics in 2028 be hosted? What is the population of that country raised to the 0.43 power?"


# ### Define Search and LLMs

# In[5]:


search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)


# ### Tools

# In[6]:


tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="useful for when you need to look up facts and statistics",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]


# ### Model, Planner, Executor, and Agent

# In[7]:


## Planner and Executor should use memory, i.e., chat models to leverage previous information


# In[8]:


model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(prompt)

