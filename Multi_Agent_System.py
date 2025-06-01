from modules.Self_Corrective_RAG_Agent_module import build_self_corrective_rag_agent
Self_Corrective_RAG_Agent = build_self_corrective_rag_agent()
# Self_Corrective_RAG_Agent = Self_Corrective_RAG_Agent.compile()

# # example usage
# VERBOSE = True
# inputs = {"messages": [("human", "explain uncertainty principle in quantum mechanics")]}
# for output in Self_Corrective_RAG_Agent.stream(inputs):
#     print("\n---\n")




import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

# Access API keys and credentials
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
TIMESCALE_DB_URI  = os.environ["TIMESCALE_DB_URI"]
TAVILY_API_KEY    = os.environ["TAVILY_API_KEY"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"] 
MAIN_AGENT_DB_URI = os.environ["MAIN_AGENT_DB_URI"]

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"]    = "langchain-academy"




# Calculators
from modules.calculators_module import arithmetic_calculator
# # example usage
# arithmetic_calculator(2, 3, 'add')




# Long Term memory in database
from modules.Long_Term_Memory_Module import langgraph_ltm_to_pydantic_ltm, replace_ltm_in_db, SessionLocal, LTM_Table_Skeleton





# -------------------------
# IMPORTS
# -------------------------

from IPython.display import Image, display
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from pydantic import BaseModel, Field
from trustcall import create_extractor

from langchain_core.documents import Document
from langchain_core.messages import merge_message_runs

import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from typing import Optional, TypedDict, Literal
from datetime import datetime

from modules.configuration import Configuration


# -------------------------
# DATABASE SETUP
# -------------------------

# Connection settings for PostgreSQL (used for cross-thread memory)
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

# Initialize a persistent connection pool for efficient DB access
pool = ConnectionPool(conninfo=MAIN_AGENT_DB_URI, max_size=20, kwargs=connection_kwargs)

# Create a Postgres-backed checkpointer to persist state
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Ensures that the necessary tables are created


# -------------------------
# GRAPH STATE SCHEMA
# -------------------------

class GraphState(MessagesState):
    """Defines the main data structure used for graph state management."""
    question:            str
    documents:           list[Document]#Fuser
    candidate_answer:    str
    retries:             int
    web_fallback:        bool
    scrag_tool_call_id:  str


# -------------------------
# MODEL SETUP
# -------------------------

# Initialize OpenAI's GPT model
model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)

# In-memory store for within-thread memory (temporary and session-limited)
across_thread_memory = InMemoryStore()


# -------------------------
# SYSTEM PROMPT FOR MODEL
# -------------------------

MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. 

You have a long-term memory which stores general user information.

<user_profile>
{user_profile}
</user_profile>

Instructions:

1. Analyze the user's message carefully.

2. If personal information is present, update the user profile via `Routing_Decision` with `update_type='prof'`.

3. Do not inform the user if you updated the profile unless they explicitly ask for it.

4. If the user's name is known, use it directly without updating the profile.

5. For calculations, call `Routing_Decision` with `update_type='calc'`.

6. For scientific queries, invoke `Routing_Decision` with `update_type='rag'`.

7. After the RAG tool's message, just reply: "I hope this answers your question."

8. Verify each calculation by calling `Routing_Decision` with `update_type='calc'`. Perform all steps by calling the calc tool.
"""


class Routing_Decision(TypedDict):
    """Specifies the type of action the assistant should take."""
    update_type: Literal['prof', 'calc', 'rag']


# -------------------------
# FUNCTION: Call model
# -------------------------

def call_model(state: GraphState, config: RunnableConfig, store: BaseStore):
    """
    Loads the user's long-term memory from the store and generates a personalized response 
    using the system prompt and conversation history.
    """
    # Get the user ID from the config
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)

    # Retrieve existing long-term memory records from the database
    with SessionLocal() as db:
        user_records = db.query(LTM_Table_Skeleton).filter(LTM_Table_Skeleton.user_id == user_id).all() ######################################################

    # Set and store retrieved memory (if any)
    current_user_record = None if len(user_records) == 0 else user_records[0]
    if current_user_record:
        store.put(namespace=namespace, key=current_user_record.memory_key, value=current_user_record.memory_value)

    # Search the memory store for user profile data
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Format and invoke the model
    system_msg = MODEL_SYSTEM_MESSAGE.format(user_profile=user_profile)
    response = model.bind_tools([Routing_Decision], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    return {'messages': [response]}


# -------------------------
# FUNCTION: Perform calculations
# -------------------------

def calculations(state: GraphState, config: RunnableConfig, store: BaseStore):
    """
    Invokes the appropriate calculator tool and returns the formatted result.
    """
    response = model.bind_tools([arithmetic_calculator], parallel_tool_calls=False).invoke(state["messages"][:-1])
    args = response.tool_calls[0]['args']
    result = arithmetic_calculator(args['n1'], args['n2'], args['ops'])
    content = f"{args['n1']} {args['ops']} {args['n2']} = {result}"

    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": content, "tool_call_id": tool_calls[0]['id']}]}


# -------------------------
# FUNCTION: Pass message to RAG agent
# -------------------------

def interim_rag_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    """
    Forwards the request to the self-corrective RAG agent.
    """
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "Passed on the request to self_corrective_rag_agent subgraph", "tool_call_id": tool_calls[0]['id']}]}


# -------------------------
# USER PROFILE MODEL
# -------------------------

class Profile(BaseModel):
    """Represents structured user profile information."""
    name:        Optional[str]  = Field(description="User's name",                                              default=None)
    bachelor:    Optional[str]  = Field(description="Bachelor's degree subject",                                default=None)
    master:      Optional[str]  = Field(description="Master's degree subject",                                  default=None)
    phd:         Optional[str]  = Field(description="PhD subject",                                              default=None)
    connections: list[str]      = Field(description="User's personal connections (friends, family, coworkers)", default_factory=list)
    interests:   list[str]      = Field(description="User's interests",                                         default_factory=list)


# -------------------------
# TRUSTCALL EXTRACTOR SETUP
# -------------------------

# Create a Trustcall extractor to update the user profile
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

# Instruction to guide Trustcall memory extraction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. Specifically, extract the subject in which the user earned their bachelor's degree.

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""


# -------------------------
# FUNCTION: Update user profile
# -------------------------

def update_profile(state: GraphState, config: RunnableConfig, store: BaseStore):
    """
    Uses the Trustcall extractor to analyze chat history and update the user's long-term memory.
    """
    # Get the user ID from the config
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)

    # Retrieve current user record (if any) and load it into the memory store
    with SessionLocal() as db:
        user_records = db.query(LTM_Table_Skeleton).filter(LTM_Table_Skeleton.user_id == user_id).all() ###############################################

    current_user_record = None if len(user_records) == 0 else user_records[0]
    if current_user_record:
        store.put(namespace=namespace, key=current_user_record.memory_key, value=current_user_record.memory_value)

    # Prepare existing memory for the extractor
    existing_items = store.search(namespace)
    existing_memories = [(item.key, "Profile", item.value) for item in existing_items] if existing_items else None

    # Merge messages and format instruction
    updated_messages = list(merge_message_runs(messages=[
        SystemMessage(content=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat()))
    ] + state["messages"][:-1]))

    # Run the extractor
    result = profile_extractor.invoke({"messages": updated_messages, "existing": existing_memories})

    # Store updated memories
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, rmeta.get("json_doc_id", str(uuid.uuid4())), r.model_dump(mode="json"))

    # Persist the memory to the database
    current_pydantic_ltm = langgraph_ltm_to_pydantic_ltm(store.search(namespace)[0])
    with SessionLocal() as db:
        replace_ltm_in_db(current_pydantic_ltm, db)

    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]['id']}]}


# -------------------------
# FUNCTION: Routing logic
# -------------------------

def route_message(state: GraphState, config: RunnableConfig, store: BaseStore) -> Literal['calculations', 'update_profile', 'self_corrective_rag_agent', END]:
    """
    Determines the next action based on tool call arguments:
    - Profile update
    - Calculation
    - RAG query
    """
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END

    tool_call = message.tool_calls[0]
    action = tool_call['args']['update_type']

    if action == "prof":
        return "update_profile"
    elif action == "calc":
        return "calculations"
    elif action == "rag":
        return "interim_rag_node"
    else:
        raise ValueError(f"Unexpected update_type: {action}")


# -------------------------
# BUILD THE GRAPH
# -------------------------

builder = StateGraph(GraphState, config_schema=Configuration)

builder.add_node("call_model", call_model)
builder.add_node("calculations", calculations)
builder.add_node("update_profile", update_profile)
builder.add_node("interim_rag_node", interim_rag_node)
builder.add_node("self_corrective_rag_agent", Self_Corrective_RAG_Agent.compile())

builder.add_edge(START, "call_model")
builder.add_edge("interim_rag_node", "self_corrective_rag_agent")
builder.add_edge("self_corrective_rag_agent", END)
builder.add_edge("calculations", "call_model")
builder.add_edge("update_profile", "call_model")

builder.add_conditional_edges('call_model', route_message, ['calculations', 'update_profile', "interim_rag_node", END])


# Compile the graph with persistent checkpointer and in-memory store
graph = builder.compile() #checkpointer=checkpointer, # store=across_thread_memory


