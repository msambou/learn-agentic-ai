#!/usr/bin/env python
# coding: utf-8

# In[22]:


from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from dotenv import load_dotenv
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
# from ragas.metrics import LLMContextRecall
from ragas.metrics import ContextRecall, Faithfulness, AgentGoalAccuracyWithReference, ToolCallAccuracy
from ragas import messages as ragas_messages
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.dataset_schema import MultiTurnSample
# from ragas.metrics import ToolCallAccuracy
# from ragas.metrics import AgentGoalAccuracyWithReference



import asyncio
import os

from ragas.llms import llm_factory
from openai import AsyncOpenAI


# In[2]:


load_dotenv()


# **Simple RAG**

# In[3]:


documents = [
    Document(
        page_content="Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
        metadata={"company":"Meta", "topic": "llama"}
    ),
    Document(
        page_content="Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
        metadata={"company":"Nvidia", "topic": "acquisition"}
    ),
    Document(
        page_content="Google is bringing Gemini to all older Pixel Buds",
        metadata={"company":"Google", "topic": "gemini"}
    ),
    Document(
        page_content="The first Intel Battlmage GPU benchmarks have leaked",
        metadata={"company":"Intel", "topic": "gpu"}
    ),
    Document(
        page_content="Dell partners with Nvidia to accelerate AI adoption in telecoms",
        metadata={"company":"Dell", "topic": "partnership"}
    ),
]

ids = ["id1", "id2", "id3", "id4", "id5"]


# In[4]:

embeddings_fn = OpenAIEmbeddings (
    model="text-embedding-3-large",
    openai_api_base="https://openai.vocareum.com/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

vector_store = Chroma(
    collection_name="udacity",
    embedding_function=embeddings_fn,
)

vector_store.add_documents(documents=documents, ids=ids)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})


# In[5]:


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openai.vocareum.com/v1",
)

ragas_llm = llm_factory("gpt-4o-mini", client=client)



# In[6]:


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}

    Question: {query}
    """
)

chain = prompt | llm | StrOutputParser()


# In[7]:


def format_docs(relevant_docs):
    return "\n".join(doc.page_content for doc in relevant_docs)


# In[8]:


query = "Who is partnering with Nvidia?"
relevant_docs = retriever.invoke(query)
chain.invoke({"context": format_docs(relevant_docs), "query": query})


# **Ragas Evaluation**

# In[9]:


sample_queries = [
    "What is Meta's latest development in AI?",
    "Which company did Nvidia acquire?",
    "What AI feature is Google adding to older Pixel Buds?",
    "What recent information has leaked about Intel GPUs?",
    "Which company did Dell partner with to accelerate AI adoption?"
]


# In[10]:


expected_responses = [
    "Meta drops multimodal Llama 3.2 — here's why it's such a big deal",
    "Chip giant Nvidia acquires OctoAI, a Seattle startup that helps companies run AI models",
    "Google is bringing Gemini to all older Pixel Buds",
    "The first Intel Battlemage GPU benchmarks have leaked",
    "Dell partners with Nvidia to accelerate AI adoption in telecoms"
]


# In[11]:


dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = chain.invoke(
        {
            "context": format_docs(relevant_docs), 
            "query": query
        }
    )
    # dataset.append(
    #     {
    #         "user_input": query,
    #         "retrieved_contexts": [doc.page_content for doc in relevant_docs],
    #         "response": response,
    #         "reference": reference,
    #     }
    # )
    dataset.append(
        {
            "question": query,
            "contexts": [doc.page_content for doc in relevant_docs],  # list[str]
            "answer": response,
            "ground_truths": [reference],  # list[str]
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)


# In[12]:


evaluator_llm = LangchainLLMWrapper(ragas_llm)

# result = evaluate(
#     dataset=evaluation_dataset,
#     metrics=[ContextRecall(llm=ragas_llm), Faithfulness(llm=ragas_llm), FactualCorrectness(llm=ragas_llm)],
#     llm=evaluator_llm,
# )

context_metric = ContextRecall(llm=ragas_llm)

# result =  context_metric.score(
#              user_input="What is the capital of France?",
#              retrieved_contexts=["Paris is the capital of France."],
#              reference="Paris is the capital and largest city of France."
#          )

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[context_metric],
    llm=evaluator_llm,
)


# In[13]:


# result


# # **LangGraph workflow**

# # In[14]:


# pokemon_types_map = {
#     "pikachu": "electric",
#     "eevee": "normal",
#     "bulbasaur": "grass/poison",
#     "squirtle": "water",
#     "charizard": "fire/flying",
#     "jigglypuff": "normal/fairy",
#     "meowth": "normal",
#     "psyduck": "water",
#     "machamp": "fighting",
#     "gengar": "ghost/poison",
#     "alakazam": "psychic",
#     "snorlax": "normal",
#     "dragonite": "dragon/flying",
# }


# # In[15]:


# @tool
# def get_pokemon_type(pokemon_name: str) -> str:
#     """Fetches the type of the specified Pokémon.

#     Args:
#         pokemon_name : The name of the Pokémon (e.g., 'pikachu', 'charizard', 'eevee').

#     Returns:
#         str: The type(s) of the Pokémon.

#     Raises:
#         KeyError: If the specified Pokémon is not found in the data source.
#     """
#     try:
#         pokemon_name = pokemon_name.lower().strip()
#         if pokemon_name not in pokemon_types_map:
#             raise KeyError(
#                 f"Pokémon '{pokemon_name}' not found. Available Pokémon: {', '.join(pokemon_types.keys())}"
#             )
#         return pokemon_types_map[pokemon_name]
#     except Exception as e:
#         raise Exception(f"Error fetching Pokémon type: {str(e)}")


# # In[16]:


# tools = [get_pokemon_type]


# # In[17]:


# llm_with_tools = llm.bind_tools(tools)


# # In[18]:


# def agent(state: MessagesState):
#     ai_message = llm_with_tools.invoke(state["messages"])
#     return {"messages": ai_message}


# # In[19]:


# def router(state: MessagesState):
#     last_message = state["messages"][-1]
#     if last_message.tool_calls:
#         return "tools"

#     return END


# # In[20]:


# workflow = StateGraph(MessagesState)

# workflow.add_node("agent", agent)
# workflow.add_node("tools", ToolNode(tools))

# workflow.add_edge(START, "agent")

# workflow.add_conditional_edges(
#     source="agent", 
#     path=router, 
#     path_map=["tools", END]
# )

# workflow.add_edge("tools", "agent")


# # In[21]:


# graph = workflow.compile()

# display(
#     Image(
#         graph.get_graph().draw_mermaid_png()
#     )
# )


# # In[23]:


# result = graph.invoke(
#     {"messages": [HumanMessage(content="What is the Gengar's type?")]}
# )


# # In[24]:


# for message in result["messages"]:
#     message.pretty_print()


# # **Converting to Ragas**

# # In[25]:


# ragas_trace = convert_to_ragas_messages(result["messages"])


# # In[26]:


# ragas_trace


# # **Evaluate Tool Use**

# # In[27]:


# sample = MultiTurnSample(
#     user_input=ragas_trace,
#     reference_tool_calls=[
#         ragas_messages.ToolCall(
#             name="get_pokemon_type", 
#             args={"pokemon_name": "gengar"}
#         )
#     ],
# )


# # In[28]:


# scorer = ToolCallAccuracy()
# scorer.llm = llm


# # In[29]:


# scorer.multi_turn_ascore(sample)


# # **Evaluate Agent Goal**

# # In[30]:


# sample = MultiTurnSample(
#     user_input=ragas_trace,
#     reference="What is the Gengar's type?",
# )


# # In[31]:


# scorer = AgentGoalAccuracyWithReference()
# scorer.llm = LangchainLLMWrapper(llm)


# # In[32]:


# scorer.multi_turn_ascore(sample)


# # In[ ]:
