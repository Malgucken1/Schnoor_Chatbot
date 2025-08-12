import os
from dotenv import load_dotenv
import streamlit as st

# langchain imports
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from supabase.client import Client, create_client
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

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
prompt = ChatPromptTemplate.from_messages([
    ("system", """Du bist Schnoor, ein hilfreicher Chatbot.
Du solltest daher zuerst das Tool 'retrieve' verwenden, um relevante Informationen aus der Supabase-Datenbank zu holen. 
- Wenn passende Informationen gefunden werden, nutze diese zur Beantwortung.
- Wenn keine passenden Informationen gefunden werden, dann kannst du dein eigenes Wissen nutzen.
Gib, wenn möglich, am Ende immer die Quellen an, die aus der Datenbank kommen.
"""),
    ("user", "{input}")
])

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

    sources = []
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        title = meta.get("title", "Unbekannter Titel")
        page = meta.get("page", "?")
        file_source = meta.get("source", "Unbekannte Datei")
        sources.append(f"{title} – Seite {page} – {file_source}")

    return content, {"sources": sources}

tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI Setup
st.set_page_config(page_title="Schnoor - Agentic RAG Chatbot", page_icon="🤖")
st.title("🤖 Schnoor´s Chatbot")

# SESSION STATE INITIALIZATION
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

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
        new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name

    st.markdown("---")

    uploaded_file = st.file_uploader("Datei hochladen", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
        except Exception:
            content = "<Datei konnte nicht als Text gelesen werden>"
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

def format_chat_history(messages):
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

if user_question:
    st.session_state.chats[st.session_state.current_chat].append(HumanMessage(user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

    chat_history_str = format_chat_history(st.session_state.chats[st.session_state.current_chat])

    with st.spinner("Agent antwortet..."):
        result = agent_executor.invoke({
            "input": user_question})
        })


    ai_message = result["output"]
    with st.chat_message("assistant"):
        sources = result.get("artifacts", {}).get("sources", [])
        if sources:
            quelle_html = "<br>".join(
                f"<span style='color:gray; font-size:small; cursor:help;' title='{src}'>Quelle</span>"
                for src in sources
            )
            st.markdown(f"{ai_message}<br><br>{quelle_html}", unsafe_allow_html=True)
        else:
            st.markdown(ai_message, unsafe_allow_html=True)

    st.session_state.chats[st.session_state.current_chat].append(AIMessage(ai_message))
