import streamlit as st
import os
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
api_key = st.secrets["google"]["api_key"]

# Streamlit app setup
st.set_page_config(page_title="Sunny's Chatbot", layout="centered")
st.title("🤖 Sunny's Gemini Chatbot")

# Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=api_key
)

# Initialize memory & conversation
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )
    st.session_state.chat_history = []
    st.session_state.pdf_context = ""  # store raw pdf content

# Upload PDF
pdf_file = st.file_uploader("📄 Upload a PDF to summarize", type=["pdf"])

if pdf_file:
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    # Save to memory for future QA
    st.session_state.pdf_context = text
    st.session_state.memory.chat_memory.add_user_message("This is the content of the uploaded PDF:\n" + text)
    st.session_state.memory.chat_memory.add_ai_message("Understood. I'll use this document to answer related questions.")

    if st.button("🧠 Summarize PDF Professionally"):
        with st.spinner("Summarizing..."):
            prompt_template = PromptTemplate(
                input_variables=["text"],
                template="""
You are a professional technical writer. Summarize the following content into a single, clear, formal paragraph:

{text}
"""
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            summary = chain.run(text=text)

            st.success("✅ Summary:")
            st.markdown(summary)

# Chat input
user_input = st.chat_input("💬 Ask me anything about the document or anything else...")

if user_input:
    response = st.session_state.conversation.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat
for speaker, message in st.session_state.chat_history:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.markdown(message)
