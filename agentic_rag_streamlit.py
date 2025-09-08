# import basics
import os
from dotenv import load_dotenv
import io
from PyPDF2 import PdfReader
import docx

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# import supabase db
from supabase.client import Client, create_client

# --- Streamlit Config MUSS ganz oben stehen ---
st.set_page_config(page_title="Schnoor - Agentic RAG Chatbot", page_icon="🤖")

# load environment variables
load_dotenv()  

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiating embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
# initiating llm
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        # Hier das Wort "Quelle" mit grauer, kleiner Schrift und Tooltip mit PDF-Namen:
        f"Content: {doc.page_content}\n\n"
        f"<span style='color:gray; font-size:small; cursor:help;' title='{doc.metadata.get('source', 'Unbekannte Quelle')}'>Quelle</span>"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------- Passwortabfrage ----------------
APP_PASSWORD = os.environ.get("APP_PASSWORD")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Passwortgeschützt")
    password_input = st.text_input("Bitte Passwort eingeben:", type="password")

    if st.button("Login"):
        if password_input == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("Erfolgreich eingeloggt!")
            st.rerun()
        else:
            st.error("Falsches Passwort!")

    st.stop()
# -------------------------------------------------

# initiating streamlit app
st.title("🤖 Schnoor´s - Agentic RAG Chatbot")
st.markdown(
    "<small>Du willst wissen, woher die Information stammt? Dann frage nach dem Dokumentennamen.</small>",
    unsafe_allow_html=True
)


# SESSION STATE INITIALIZATION
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Neuer Chat"
if "chats" not in st.session_state:
    st.session_state.chats = {"Neuer Chat": []}

# ----- Hilfsfunktion für Chat-Titel -----
def generate_chat_title(first_prompt: str) -> str:
    response = llm.predict(
        f"Erzeuge einen kurzen, prägnanten Titel für einen Chat basierend auf diesem Text: '{first_prompt}'"
    )
    title = response.strip().strip('"').strip("'")
    if not title:
        title = f"Chat {len(st.session_state.chats) + 1}"
    return title

# ----- Sidebar: Chat-Auswahl und Datei-Upload -----
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
        # Automatisch Chat-Titel erstellen, wenn "Neuer Chat"
        if st.session_state.current_chat == "Neuer Chat" or st.session_state.current_chat.endswith("(neuer Chat)"):
            new_title = generate_chat_title(file_content)
            st.session_state.chats[new_title] = []
            st.session_state.current_chat = new_title

        st.session_state.chats[st.session_state.current_chat].append(HumanMessage(file_content))
        st.success("Datei-Inhalt zum Chat hinzugefügt!")

# ----- Chat-Verlauf anzeigen -----
for message in st.session_state.chats[st.session_state.current_chat]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content, unsafe_allow_html=True)

# ----- User Input -----
user_question = st.chat_input("Frag mich was!")

if user_question:
    current = st.session_state.current_chat
    # Automatische Titelgenerierung beim ersten User-Eingang
    if current == "Neuer Chat" or current.endswith("(neuer Chat)"):
        new_title = generate_chat_title(user_question)
        st.session_state.chats[new_title] = st.session_state.chats.pop(current)
        st.session_state.current_chat = new_title
        current = new_title

    st.session_state.chats[current].append(HumanMessage(user_question))
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # --- Agentaufruf ---
    with st.spinner("Agent antwortet..."):
        result = agent_executor.invoke({
            "input": user_question,
            "chat_history": st.session_state.chats[current]
        })

    ai_message = result["output"]

    with st.chat_message("assistant"):
        st.markdown(ai_message, unsafe_allow_html=True)
        st.session_state.chats[current].append(AIMessage(ai_message))
