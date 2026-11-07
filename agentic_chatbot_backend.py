from  langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages  import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message  import add_messages

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    #take user query from state
    messages = state['messages']
    # send  to llm
    response = model.invoke(messages)
    # response store state
    return {'messages': [response]}


checkpoint = MemorySaver()

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


chatbot = graph.compile(checkpointer=checkpoint)