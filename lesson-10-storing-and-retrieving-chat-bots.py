#!/usr/bin/env python
# coding: utf-8

# # Storing and Retrieving Chat History

# ### Imports

# In[1]:


import os
#os.environ["OPENAI_API_KEY"] = "..."


# In[3]:


from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain import OpenAI, ConversationChain


# ### Create Chat Message History

# In[4]:


history = ChatMessageHistory()
history.add_user_message("hello! let's talk about giraffes")
history.add_ai_message("hi! i'm down to talk about giraffes")


# In[5]:


history


# In[6]:


dicts = messages_to_dict(history.messages)


# In[7]:


dicts


# ### Chat History with New Messages

# In[8]:


new_messages = messages_from_dict(dicts)


# In[9]:


new_messages


# In[10]:


llm = OpenAI(temperature=0)
history = ChatMessageHistory(messages=new_messages)
buffer = ConversationBufferMemory(chat_memory=history)


# In[11]:


history


# In[12]:


buffer


# ### Conversation with Chat History (Buffer)

# In[13]:


conversation = ConversationChain(llm=llm, memory=buffer, verbose=True)
print(conversation.predict(input="what are they?"))


# In[14]:


conversation.memory.chat_memory.messages

