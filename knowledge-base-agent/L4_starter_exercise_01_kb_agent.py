#!/usr/bin/env python
# coding: utf-8

# # Exercise - Knowledge Base Agent - STARTER

# In this exercise, you’ll build a Knowledge Base Agent using LangGraph, which can:
# 
# Efficiently process long documents using text embedding and chunking.
# Retrieve information from a vector database.
# Augment user queries with retrieved contextual documents.
# Generate accurate responses using an LLM.
# 

# **Challenge**

# Your task is to create a LangGraph Workflow that includes:
# 
# - A document loading and vectorization process for a knowledge base.
# - An Agent Node capable of:
#     - Retrieving relevant knowledge.
#     - Augmenting responses with contextual documents.
#     - Generating accurate answers.
# - Conditional routing to control query resolution.
# - Optimization techniques such as text chunking and embedding search.

# By the end of this exercise, you’ll have built an AI-powered Knowledge Base Agent that uses a structured process to generate accurate answers.
# 
# 

# ## 0. Import the necessary libs

# In[ ]:


from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from IPython.display import Image, display


import os
import asyncio

from dotenv import load_dotenv
load_dotenv()



# ## 1. Instantiate Chat Model with your API Key

# To be able to connect with OpenAI, you need to instantiate an ChatOpenAI client passing your OpenAI key.
# 
# You can pass the `api_key` argument directly.
# ```python
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.0,
#     api_key="voc-",
# )
# ```

# In[ ]:


# TODO - Instantiate your chat model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# In[ ]:


# TODO - Instantiate your embeddings model
embeddings_fn = OpenAIEmbeddings (
    model="text-embedding-3-large",
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ## 2. Load and Process Documents

# In[ ]:


# TODO create your Chroma vector store with a collection name 
# and the embedding function
vector_store = Chroma(
    collection_name="udacity",
    embedding_function=embeddings_fn
)


# In[ ]:


file_path = "files/compact-guide-to-large-language-models.pdf"


# In[ ]:


loader = PyPDFLoader(file_path)


# In[ ]:


async def load_pages(loader):
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

pages = asyncio.run(load_pages(loader))
# In[ ]:


# TODO - Create a text splitter with chunk_size and chunk_overlap 
# values of 1000 and 200, respectively
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)


# In[ ]:


all_splits = text_splitter.split_documents(pages)


# In[ ]:


_ = vector_store.add_documents(documents=all_splits)


# ## 3. Define State Schema

# We define a State Schema for managing:
# 
# - User query
# - Retrieved documents
# - Generated answer

# In[ ]:


# TODO - Create your state schema named State inheriting from MessagesState
# with question(str), documents(List) and answer(str) attributes
class State(MessagesState):
    question: str
    documents: List[str]
    answer: str



# ## 4. RAG Nodes

# The agent should:
# - fetch relevant document chunks based on the user query
# - combine the retrieved documents and use them as context
# - invoke the LLM to generate a response

# In[ ]:


def retrieve(state: State):
    print("\n*************** In Retrieve **********************")
    question = state["question"]

    # TODO - Use the vector store to retrieve similar documents to the question
    # Use the similarity_search() method
    retrieved_docs = vector_store.similarity_search(query=question)

    return {"documents": retrieved_docs}


# In[ ]:


def augment(state: State):
    print("\n*************** In Augment **********************")
    question = state["question"]
    documents = state["documents"]
    docs_content = "\n\n".join(doc.page_content for doc in documents)

    # TODO - Create a RAG ChatPromptTemplate with question and context variables
    template = ChatPromptTemplate.from_template("""
You are an expert at answering questions.
Give a detailed answer to the question {question}

If you don't know the answer, say I don't know.
Only use these documents as knowledge base:
{context}
""")

    messages = template.invoke(
        {"context": docs_content, "question": question}
    ).to_messages()

    return {"messages": messages}


# In[ ]:


def generate(state: State):
    # TODO - Invoke the LLM passing the messages from state
    print("\n*************** In generate **********************")
    messages = state.get('messages', [])
    ai_message = llm.invoke(messages)
    print("\n*************** End generate **********************")
    return {"answer": ai_message.content, "messages": ai_message}


# ## 5. Build the LangGraph Workflow

# In[ ]:


workflow = StateGraph(State)
# TODO - add all the nodes and edges
workflow.add_node("retrieve", retrieve)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "augment")
workflow.add_edge("augment", "generate")
workflow.add_edge("generate", END)

# In[ ]:


graph = workflow.compile()

display(
    Image(
        graph.get_graph().draw_mermaid_png()
    )
)


# ## 6. Invoke the Agent with a Query

# Run and Print the retrieved documents to check search accuracy.

# In[ ]:


output = graph.invoke(
    {"question": "What are Open source models?"}
)


# In[ ]:

# print(output)
# output["answer"]


# In[ ]:


for message in output["messages"]:
    message.pretty_print()


# ## 7. Experiment

# Now that you understood how it works, experiment with new things.
# 
# - Change the embedding model
# - Change the parameters of RecursiveCharacterTextSplitter(chunk_size and chunk_overlap)
# - Use your own document
# - Add More File Types
