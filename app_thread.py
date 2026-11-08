from agentic_chatbot_backend import chatbot, model
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import uuid
import re

st.title("Agentic Chatbot with LangGraph")

DEFAULT_TITLE = "New chat"


# ========================= Helper functions ================================

# Generate a unique thread ID for each new conversation
# (the ID stays a UUID internally; the title is only for display)
def generate_thread_id() -> str:
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id: str):
    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        st.session_state["thread_titles"][thread_id] = DEFAULT_TITLE


# Create a completely new chat conversation
def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


# Fallback title: first few words of the user's message
def fallback_title(text: str) -> str:
    words = text.strip().split()
    title = " ".join(words[:6])
    return title + ("..." if len(words) > 6 else "") if title else DEFAULT_TITLE


# Generate a short title from the first user message
def generate_title(first_message: str) -> str:
    try:
        prompt = (
            "Write a concise title (maximum 5 words) for a conversation that "
            "starts with the message below. Reply with the title only, with "
            "no quotes or punctuation at the end.\n\n"
            f"Message: {first_message}"
        )
        response = model.invoke(prompt)
        title = response.content if isinstance(response.content, str) else ""
        # Qwen models may include <think>...</think> reasoning; remove it
        title = re.sub(r"<think>.*?</think>", "", title, flags=re.DOTALL)
        title = title.strip().strip('"').strip("'").rstrip(".")
        return title[:50] if title else fallback_title(first_message)
    except Exception:
        return fallback_title(first_message)


# Load a past conversation's messages from LangGraph's memory
def load_conversation(thread_id: str) -> list:
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state and state.values else []

    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            # Skips AI messages that only contain tool calls
            history.append({"role": "assistant", "content": msg.content})
    return history


# ========================= Session state setup ================================

# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# Map each thread ID to its display title
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# Create the current thread ID (must be a string, not a list)
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])


# ========================= Sidebar threading feature ================================

# Display the sidebar title
st.sidebar.title("My Conversations")

if st.sidebar.button("New chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# List all conversations by title, newest first
for thread_id in reversed(st.session_state["chat_threads"]):
    title = st.session_state["thread_titles"].get(thread_id, DEFAULT_TITLE)
    is_current = thread_id == st.session_state["thread_id"]

    if st.sidebar.button(
        title,
        key=f"thread_{thread_id}",
        use_container_width=True,
        type="primary" if is_current else "secondary",
    ):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.rerun()


# ========================= Display existing chat history ================================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


# ========================= Chat input and response ================================

# Create the chat input box
user_input = st.chat_input("Type here")

# Run this block after the user submits a message
if user_input:

    current_thread = st.session_state["thread_id"]
    is_first_message = st.session_state["thread_titles"].get(current_thread) == DEFAULT_TITLE

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": current_thread
        }
    }

    # Create the assistant chat-message container
    with st.chat_message("assistant"):

        # Stream the assistant response token by token
        ai_message = st.write_stream(

            # Return only the content of AI message chunks
            message_chunk.content

            # Stream messages from the LangGraph chatbot
            for message_chunk, metadata in chatbot.stream(
                {
                    # Send the latest user message to the chatbot
                    "messages": [HumanMessage(content=user_input)]
                },

                # Use the current conversation thread
                config=CONFIG,

                # Stream individual message chunks
                stream_mode="messages"
            )

            # Display only AI messages with plain-string content
            # This prevents tool and user messages from appearing
            if isinstance(message_chunk, AIMessage)
            and isinstance(message_chunk.content, str)
        )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # Name the conversation after its first message, then refresh the sidebar
    if is_first_message:
        st.session_state["thread_titles"][current_thread] = generate_title(user_input)
        st.rerun()