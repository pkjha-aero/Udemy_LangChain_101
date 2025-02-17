#!/usr/bin/env python
# coding: utf-8

# # Memory and Chat Bots

# ### Imports

# In[1]:


import os
os.environ["OPENAI_API_KEY"] = "..."


# In[2]:


from langchain import OpenAI, ConversationChain


# ### Printing Predictions

# In[3]:


llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)


# In[4]:


print(conversation.predict(input="Hi there!"))
print(conversation.predict(input="Can we talk about weather?"))
print(conversation.predict(input="It's a beautiful day today"))


# ### Creating an interactive terminal Chat Bot

# In[5]:


llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm)


# In[6]:


print("Welcome to your AI chatbot! What's on your mind?")


# In[7]:


for _ in range(0, 3):
    human_input = input("You: ")
    ai_response = conversation.predict(input=human_input)
    print(f"AI: {ai_response}")

