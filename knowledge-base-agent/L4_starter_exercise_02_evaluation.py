#!/usr/bin/env python
# coding: utf-8

# # Exercise - Knowledge Base Agent - STARTER

# In this exercise, you’ll implement and evaluate a RAG (Retrieval-Augmented Generation) pipeline, using RAGAS metrics and MLflow for logging the process.
# 

# **Challenge**

# Your task is to create a LangGraph Workflow that includes:
# 
# - A RAG pipeline for information retrieval.
# - An LLM-based judge for evaluation.
# - RAGAS metrics for quality assessment.
# - MLflow logging for observability.
# 
# The workflow should:
# 
# - Retrieve, augment, and generate answers.
# - Evaluate the answers using RAGAS.
# - Log performance metrics in MLflow.

# ## 0. Import the necessary libs

# In[ ]:


import mlflow
from mlflow import log_params, log_metrics
from typing import List, Dict, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langchain_classic.prompts import ChatPromptTemplate
from ragas import evaluate
from datasets import Dataset
from IPython.display import Image, display

import os


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


from dotenv import load_dotenv
load_dotenv()


# In[ ]:


# TODO - Instantiate your chat model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# In[ ]:


# TODO - Instantiate your llm as judge model
# This will evaluate the responses
llm_judge = ChatOpenAI(
    model="gpt-4o",
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



# ## 2. MLFlow

# In[ ]:


mlflow.set_experiment("udacity")


# # In[ ]:


# with mlflow.start_run(run_name="l4_exercise_02") as run:
# log_params(
#         {
#             "embeddings_model":embeddings_fn.model,
#             "llm_model": llm.model_name,
#             "llm_judge_model": llm_judge.model_name,
#         }
#     )
# print(run.info)


# # In[ ]:


# mlflow_run_id = run.info.run_id


# # In[ ]:


# mflow_client = mlflow.tracking.MlflowClient()


# # In[ ]:


# mflow_client.get_run(mlflow_run_id)


# # ## 3. Load and Process Documents

# # In[ ]:


# # Initialize vector store
vector_store = Chroma(
    collection_name="udacity",
    embedding_function=embeddings_fn
)

# Load and process PDF documents
file_path = "compact-guide-to-large-language-models.pdf"
loader = PyPDFLoader(file_path)

pages = []
for page in loader.load():
    pages.append(page)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(pages)

# Store document chunks in the vector database
_ = vector_store.add_documents(documents=all_splits)


# # ## 4. Define State Schema

# We define a State Schema for managing:
# 
# - MLFlow Run id
# - User query
# - Ground Truth
# - Retrieved documents
# - Generated answer
# - Evaluation Report

# run_id(str), ground_truth(str), evaluation(Dict),vquestion(str), documents(List) and answer(str)

# # In[ ]:


# TODO - Create your state schema
class State(MessagesState):
    



# # ## 5. RAG Nodes

# # The agent should:
# # - fetch relevant document chunks based on the user query
# # - combine the retrieved documents and use them as context
# # - invoke the LLM to generate a response
# # - evaluate the pipeline based on the ground_truth

# # In[ ]:


# def retrieve(state: State):
#     question = state["question"]
#     retrieved_docs = vector_store.similarity_search(question)
#     return {"documents": retrieved_docs}


# # In[ ]:


# def augment(state: State):
#     question = state["question"]
#     documents = state["documents"]
#     docs_content = "\n\n".join(doc.page_content for doc in documents)

#     template = ChatPromptTemplate([
#         ("system", "You are an assistant for question-answering tasks."),
#         ("human", "Use the following pieces of retrieved context to answer the question. "
#                 "If you don't know the answer, just say that you don't know. " 
#                 "Use three sentences maximum and keep the answer concise. "
#                 "\n# Question: \n-> {question} "
#                 "\n# Context: \n-> {context} "
#                 "\n# Answer: "),
#     ])

#     messages = template.invoke(
#         {"context": docs_content, "question": question}
#     ).to_messages()

#     return {"messages": messages}


# # In[ ]:


# def generate(state: State):
#     ai_message = llm.invoke(state["messages"])
#     return {"answer": ai_message.content, "messages": ai_message}


# # In[ ]:


# def evaluate_rag(state: State):
#     question = state["question"]
#     documents = state["documents"]
#     answer = state["answer"]
#     ground_truth = state["ground_truth"]
#     dataset = Dataset.from_dict(
#         {
#             "question": [question],
#             "answer": [answer],
#             "contexts": [[doc.page_content for doc in documents]],
#             "ground_truth": [ground_truth]
#         }
#     )

#     evaluation_results = evaluate(
#         dataset=dataset,
#         llm=llm_judge
#     )
#     print(evaluation_results)

#     # TODO - Log metrics in MLflow
#     # The evaluation_results output value is a list
#     # Example: evaluation_results["faithfulness"][0]
#     with mlflow.start_run(state["run_id"]):
#         log_metrics({
#             "faithfulness": "",
#             "context_precision": "",
#             "context_recall": "",
#             "answer_relevancy": "",
#         })

#     return {"evaluation": evaluation_results}


# # ## 6. Build the LangGraph Workflow

# # In[ ]:


# # TODO - add all the nodes and edges
# workflow = StateGraph(State)


# # In[ ]:


# graph = workflow.compile()

# display(
#     Image(
#         graph.get_graph().draw_mermaid_png()
#     )
# )


# # ## 7. Invoke the Workflow with a Query

# # In[ ]:


# reference = [
#     {
#         "question": "What are Open source models?",
#         "ground_truth": "Open-source models are AI or machine learning " 
#                         "models whose code, architecture, and in some cases, " 
#                         "training data and weights, are publicly available for " 
#                         "use, modification, and distribution. They enable " 
#                         "collaboration, transparency, and innovation by allowing " 
#                         "developers to fine-tune, deploy, or improve them without " 
#                         "proprietary restrictions.",
#     }
# ]


# # In[ ]:


# output = graph.invoke(
#     {
#         "question": reference[0]["question"],
#         "ground_truth": reference[0]["ground_truth"],
#         "run_id": mlflow_run_id
#     }
# )


# # ## 8. Inspect in MLFlow

# # In[ ]:


# # TODO - Get MLFlow Run with .get_run()
# mflow_client


# # ## 9. Experiment

# # Now that you understood how it works, experiment with new things.
# # 
# # - Change RAG parameters: embedding model, chunk_size, chunk_overlap...
# # - Create multiple runs
# # - Improve your reference with more questions and ground_truth answers
# # - Use the results to understand what are the best parameters
# # - Create an Agent that picks the best combination
