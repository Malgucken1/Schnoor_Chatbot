import streamlit as st  # muss als allererstes Streamlit-Kommando kommen!
st.set_page_config(page_title="Schnoor - Agentic RAG Chatbot", page_icon="🤖")

import os
from dotenv import load_dotenv
import io
from PyPDF2 import PdfReader
import docx

# langchain imports
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from supabase.client import Client, create_client
from langchain_core.messages import HumanMessage, AIMessage

# load environment variables
load_dotenv()

# supabase init
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# ----- Retrieval Tool -----
@tool
def retrieve(query: str):
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Nur den Dateinamen verwenden
    sources = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "")
        if src:
            filename = os.path.basename(src)
            sources.append(filename)

    return {
        "content": serialized_content,
        "sources": sources
    }

# ----- Caching -----
@st.cache_resource
def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

@st.cache_resource
def get_agent_executor():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = [retrieve]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True), llm, prompt

# ----- Initialisierung -----
vector_store = get_vector_store()
agent_executor, llm, prompt = get_agent_executor()

# ----- Hilfsfunktion für Chat-Titel -----
def generate_chat_title(first_prompt: str) -> str:
    response = llm.predict(
        f"Erzeuge einen kurzen, prägnanten Titel für einen Chat basierend auf diesem Text: '{first_prompt}'"
    )
    title = response.strip().strip('"').strip("'")
    if not title:
        title = f"Chat {len(st.session_state.chats) + 1}"
    return title

# ----- Streamlit UI -----
st.title("🤖 Schnoor´s Chatbot")

# SESSION STATE INITIALIZATION
if "chats" not in st.session_state:
    st.session_state.chats = {"Neuer Chat": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Neuer Chat"

# SIDEBAR mit Chat-Verzeichnis und Datei-Upload
with st.sidebar:
    st.header("Chats")
    selected_chat = st.selectbox(
        "Chat auswählen",
        list(st.session_state.chats.keys()),
        index=list(st.session_state.chats.keys()).index(st.session_state.current_chat),
    )
    st.session_state.current_chat = selected_chat

    if st.button("Neuen Chat erstellen"):
        placeholder_chat_name = f"Chat {len(st.session_state.chats) + 1} (neuer Chat)"
        st.session_state.chats[placeholder_chat_name] = []
        st.session_state.current_chat = placeholder_chat_name

    st.markdown("---")
    uploaded_file = st.file_uploader("Datei hochladen", type=["txt", "pdf", "docx"])

# Datei-Inhalt lesen
file_content = None
if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            file_content = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif uploaded_file.type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]:
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            file_content = "\n".join([para.text for para in doc.paragraphs])
        else:
            file_content = "<Dateityp wird nicht unterstützt>"
    except Exception as e:
        file_content = f"<Datei konnte nicht gelesen werden: {str(e)}>"

    if st.button("Datei-Inhalt zum Chat hinzufügen") and file_content:
        st.session_state.chats[st.session_state.current_chat].append(HumanMessage(file_content))
        st.success("Datei-Inhalt zum Chat hinzugefügt!")

# Chatverlauf anzeigen
for message in st.session_state.chats[st.session_state.current_chat]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content, unsafe_allow_html=True)

# User Input
user_question = st.chat_input("Frag mich was!")

if user_question:
    current = st.session_state.current_chat
    if current == "Neuer Chat" or current.endswith("(neuer Chat)"):
        new_title = generate_chat_title(user_question)
        st.session_state.chats[new_title] = st.session_state.chats.pop(current)
        st.session_state.current_chat = new_title
        current = new_title

    st.session_state.chats[current].append(HumanMessage(user_question))
    with st.chat_message("user"):
        st.markdown(user_question)
    
# --- Agentaufruf mit unsichtbarem Prompt für Dateiname ---
with st.spinner("Agent antwortet..."):
    augmented_question = f"""
    {user_question}
    ---
    Wichtig: Gib immer den Dokumentennamen zurück,
    aus dem die Antwort stammt (nur den Dateinamen, kein Pfad, kein Link). Erfinde keine Quellen. 
    Unterlasse so etwas: Diese Informationen stammen aus der Quelle. Gib den Dokumentennamen nur zurück, wenn du den Daateinamen kennst. Ansonsten lasse es weg!
    Format: 'Quelle: <Dateiname>'
    """
    result = agent_executor.invoke({
        "input": augmented_question,
        "chat_history": st.session_state.chats[current]
    })

ai_message = result["output"]

# Quellen aus dem Retrieval nehmen (nicht aus LLM-Text)
sources = result.get("sources", [])
unique_sources = list(dict.fromkeys(filter(None, sources)))  # Duplikate entfernen

# AI-Nachricht und Quelle anzeigen
with st.chat_message("assistant"):
    st.markdown(ai_message, unsafe_allow_html=True)
    if unique_sources:
        st.markdown(f"_Quelle: {', '.join(unique_sources)}_")

# AIMessage speichern
st.session_state.chats[current].append(AIMessage(ai_message))

    
