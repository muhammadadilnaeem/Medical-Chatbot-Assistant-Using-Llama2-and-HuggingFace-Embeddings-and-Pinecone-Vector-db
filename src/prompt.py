# from langchain_core.prompts import PromptTemplate

# # Define prompt template for the model 
# prompt_template = """
# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

#rom langchain_core.prompts import PromptTemplate

# # Define enhanced prompt template for the model
# prompt_template = """
# As a professional and empathetic medical advisor, respond to the user's question using the information provided.
# If the information is insufficient, acknowledge the limitation and suggest consulting a healthcare professional.

# **Context:** 
# {context}

# **User's Question:** 
# {question}

# ---

# **Professional and Helpful Response:**
# """

from langchain_core.prompts import PromptTemplate
prompt_template = """
As a medical advisor, respond concisely and accurately to the user's question based on the provided context. If the information is insufficient, suggest consulting a healthcare professional. Use clear and structured markdown formatting to organize the information.

**Context:**
{context}

**User's Question:**
{question}

---

**Concise Response:**
"""



