import os
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

API_KEY = os.getenv("gemini_key")

if not API_KEY:
    st.error("Gemini API key not found.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = API_KEY

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Production GenAI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("⚙️ Settings")

temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.3
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    100,
    2048,
    500
)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat = None
    st.rerun()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🤖 Production Ready Gemini Chatbot")

st.caption("Powered by Gemini 2.5 Flash")

# ---------------------------------------------------
# System Prompt
# ---------------------------------------------------
SYSTEM_PROMPT = """
You are an expert Generative AI assistant.

Answer accurately.

Use bullet points whenever possible.

If you don't know the answer, say:
'I don't have enough information.'

Keep answers concise unless the user asks for details.
"""

# ---------------------------------------------------
# Gemini Client
# ---------------------------------------------------
if "client" not in st.session_state:
    st.session_state.client = genai.Client()

client = st.session_state.client

# ---------------------------------------------------
# Chat Session
# ---------------------------------------------------
if "chat" not in st.session_state or st.session_state.chat is None:

    st.session_state.chat = client.chats.create(

        model="gemini-2.5-flash",

        config=types.GenerateContentConfig(

            system_instruction=SYSTEM_PROMPT,

            temperature=temperature,

            max_output_tokens=max_tokens

        )
    )

# ---------------------------------------------------
# Chat History
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# Display Messages
# ---------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        st.caption(message["time"])

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
prompt = st.chat_input("Ask me anything...")

if prompt:

    current_time = datetime.now().strftime("%H:%M")

    st.session_state.messages.append({

        "role": "user",

        "content": prompt,

        "time": current_time

    })

    with st.chat_message("user"):

        st.markdown(prompt)

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = st.session_state.chat.send_message(prompt)

                answer = response.text

                st.markdown(answer)

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer,

            "time": datetime.now().strftime("%H:%M")

        })

        logging.info(f"User: {prompt}")

        logging.info(f"Bot: {answer}")

    except Exception as e:

        st.error("Something went wrong.")

        logging.error(str(e))

# ---------------------------------------------------
# Download Chat
# ---------------------------------------------------
if st.session_state.messages:

    conversation = ""

    for msg in st.session_state.messages:

        conversation += f"{msg['role'].upper()}: {msg['content']}\n\n"

    st.sidebar.download_button(

        "📥 Download Chat",

        conversation,

        file_name="conversation.txt"

    )