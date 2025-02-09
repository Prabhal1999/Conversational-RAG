import os
import streamlit as st
from dotenv import load_dotenv
import concurrent.futures

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Streamlit Interface
st.title("📄 Conversational RAG")
st.write("💬 Ask questions about the uploaded PDF")
st.info("Please note that the app is currently under development. Apologies for any bugs or issues.")

# Error Handling for API Key
if not groq_api_key:
    st.error("⚠️ API keys are missing from the environment.")
    st.stop()

# Initialize LLM Model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# Session ID Input
session_id = st.text_input("🆔 Enter Session ID")

# Ensure session storage for chat history
if "store" not in st.session_state:
    st.session_state.store = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Clear Cache Button
if st.button("🧹 Clear Cache"):
    st.session_state.clear() 
    st.toast("Cache cleared! Upload a new PDF.", icon="✅")

# Get Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

embeddings = get_embeddings()

# File Upload
uploaded_files = st.file_uploader("📂 Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Reset vector store and chat history when a new PDF is uploaded
    st.session_state["vectorstore"] = None
    st.session_state["chat_history"] = ChatMessageHistory()

    documents = []  

    def process_pdf(file):
        temp_pdf = f"./temp_{file.name}.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(file.getvalue())
        return PyPDFLoader(temp_pdf).load()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pdf, uploaded_files))

    for doc_list in results:
        documents.extend(doc_list)  

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Store a fresh FAISS vector store in session state
    st.session_state["vectorstore"] = FAISS.from_texts(
        [doc.page_content for doc in splits], embeddings
    ).as_retriever()

# Retrieve the current vector store
retriever = st.session_state.get("vectorstore", None)

# Contextualize Question System Prompt
contextualize_q_system_prompt = """
Look at the chat history and the latest question from the user. If the question 
depends on previous messages, rewrite it so that it makes sense on its own. 
Do not answer the question, just rewrite it if needed. If the question is 
already clear, keep it the same.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt) if retriever else None

# Answering System Prompt
system_prompt_text = """
You are an assistant that answers questions. Use the given information to answer. 
If you do not know the answer, say 'I do not know'. 
Keep your answer short (maximum 3 sentences).

{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_text),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# Question-Answering Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Final Retrieval-Augmented Generation (RAG) Chain
if retriever:
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to Manage Session Chat History
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Conversational RAG Chain with History
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat Input
    user_input = st.chat_input("✍️ Ask something about the PDF")

    if user_input:
        with st.spinner("🤔 Wait! Let me figure it out..."):
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                {"configurable": {"session_id": session_id}}
            )

        # Display Chat
        st.write("📝 **Chats:**")
        for msg in session_history.messages:
            st.write(f"{msg.type.capitalize()}: {msg.content}")


