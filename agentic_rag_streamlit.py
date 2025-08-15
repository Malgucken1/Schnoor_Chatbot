import os
from dotenv import load_dotenv
import streamlit as st
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

# embeddings & vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# llm & prompt & agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")

@tool  # ohne response_format
def retrieve(query: str):
    """
    Retrieve information related to a query from the vector store.

    Args:
        query (str): The query string.

    Returns:
        Tuple[str, dict]: Content string and dictionary with sources.
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = [doc.metadata.get("source", "Unbekannte Quelle") for doc in retrieved_docs]
    return {"content": content, "sources": sources}


tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Hilfsfunktion, um Chat-Titel aus erstem Prompt zu generieren
def generate_chat_title(first_prompt: str) -> str:
    response = llm.predict(f"Erzeuge einen kurzen, prägnanten Titel für einen Chat basierend auf diesem Text: '{first_prompt}'")
    title = response.strip().strip('"').strip("'")
    if not title:
        title = f"Chat {len(st.session_state.chats) + 1}"
    return title

# Streamlit UI Setup
st.set_page_config(page_title="Schnoor - Agentic RAG Chatbot", page_icon="🤖")
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
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/plain":
                # TXT-Dateien
                content = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                # PDF-Dateien
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                content = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        "application/msword"]:
                # DOCX-Dateien
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                content = "\n".join([para.text for para in doc.paragraphs])
            else:
                content = "<Dateityp wird nicht unterstützt>"
        except Exception as e:
            content = f"<Datei konnte nicht gelesen werden: {str(e)}>"

        st.write(f"**Datei-Inhalt (erste 500 Zeichen):**")
        st.write(content[:500])

        if st.button("Datei-Inhalt zum Chat hinzufügen"):
            st.session_state.chats[st.session_state.current_chat].append(HumanMessage(content))
            st.success("Datei-Inhalt zum Chat hinzugefügt!")

st.markdown(f"### Aktueller Chat: {st.session_state.current_chat}")

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

    with st.spinner("Agent antwortet..."):
        result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.chats[current]})

    ai_message = result["output"]
sources = result.get("artifacts", {}).get("sources", [])

# Nur eindeutige, nicht-leere Quellen
unique_sources = list(dict.fromkeys(filter(None, sources)))

with st.chat_message("assistant"):
    if unique_sources:
        # HTML für konsistente Anzeige am Ende
        quelle_html = "<br>".join(
            f"<span style='color:gray; font-size:small; cursor:help;' title='{src}'>Quelle</span>"
            for src in unique_sources
        )
        st.markdown(f"{ai_message}<br><br>{quelle_html}", unsafe_allow_html=True)
    else:
        st.markdown(ai_message, unsafe_allow_html=True)

    st.session_state.chats[current].append(AIMessage(ai_message))
