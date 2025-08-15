import os
import io
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import docx

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from supabase.client import Client, create_client
from langchain_core.messages import HumanMessage, AIMessage

# -------------------- ENV & INIT --------------------
load_dotenv()

@st.cache_resource
def init_supabase():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    return create_client(supabase_url, supabase_key)

@st.cache_resource
def init_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return SupabaseVectorStore(
        embedding=embeddings,
        client=init_supabase(),
        table_name="documents",
        query_name="match_documents",
    )

@st.cache_resource
def init_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    try:
        prompt = hub.pull("hwchase17/openai-functions-agent")
    except Exception as e:
        st.error(f"Prompt konnte nicht geladen werden: {e}")
        prompt = None
    tools = [retrieve]
    agent = create_tool_calling_agent(llm, tools, prompt) if prompt else None
    return llm, AgentExecutor(agent=agent, tools=tools, verbose=True) if agent else None

vector_store = init_vectorstore()

# -------------------- TOOLS --------------------
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    try:
        retrieved_docs = vector_store.similarity_search(query, k=2)
    except Exception as e:
        return f"Fehler bei der Suche: {e}", []

    serialized_content = "\n\n".join(
        f"Content: {doc.page_content or ''}\n\n"
        f"<span style='color:gray; font-size:small; cursor:help;' "
        f"title='{doc.metadata.get('source', 'Unbekannte Quelle')}'>Quelle</span>"
        for doc in retrieved_docs
    )
    return serialized_content, retrieved_docs

# -------------------- CHAT TITLE --------------------
def generate_chat_title(first_prompt: str) -> str:
    llm, _ = init_agent()
    try:
        response = llm.invoke(
            f"Erzeuge einen kurzen, prägnanten Titel für einen Chat basierend auf diesem Text: '{first_prompt}'"
        )
        title = getattr(response, "content", "").strip().strip('"').strip("'")
    except Exception as e:
        st.warning(f"Titelgenerierung fehlgeschlagen: {e}")
        title = ""
    return title or f"Chat {len(st.session_state.chats) + 1}"

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Schnoor - Agentic RAG Chatbot", page_icon="🤖")
st.title("🤖 Schnoor´s Chatbot")

if "chats" not in st.session_state:
    st.session_state.chats = {"Neuer Chat": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Neuer Chat"

# -------------------- SIDEBAR --------------------
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

# -------------------- DATEI-VERARBEITUNG --------------------
file_content = None
if uploaded_file:
    try:
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            file_content = "\n".join(filter(None, (page.extract_text() for page in pdf_reader.pages)))
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

# -------------------- CHATVERLAUF --------------------
for message in st.session_state.chats[st.session_state.current_chat]:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content, unsafe_allow_html=True)

# -------------------- USER INPUT --------------------
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

    _, agent_executor = init_agent()
    if agent_executor:
        try:
            with st.spinner("Agent antwortet..."):
                result = agent_executor.invoke({
                    "input": user_question,
                    "chat_history": st.session_state.chats[current]
                })

            ai_message = result.get("output", "")
            sources = result.get("sources", [])
            unique_sources = list(dict.fromkeys(filter(None, sources)))

            with st.chat_message("assistant"):
                if unique_sources:
                    quelle_html = "<br>".join(
                        f"<span style='color:gray; font-size:small; cursor:help;' title='{src}'>Quelle</span>"
                        for src in unique_sources
                    )
                    st.markdown(f"{ai_message}<br><br>{quelle_html}", unsafe_allow_html=True)
                else:
                    st.markdown(ai_message, unsafe_allow_html=True)

            st.session_state.chats[current].append(AIMessage(ai_message))
        except Exception as e:
            st.error(f"Fehler beim Abrufen der Antwort: {e}")
    else:
        st.error("Agent konnte nicht initialisiert werden.")
