from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']
    # send to llm
    response = model.invoke(messages)
    # store response in state
    return {'messages': [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    # checkpoint.list() returns newest checkpoints first.
    # A dict keeps insertion order, so the newest thread stays first.
    seen = {}
    for ck in checkpoint.list(None):
        seen.setdefault(ck.config['configurable']['thread_id'], None)

    return list(seen)