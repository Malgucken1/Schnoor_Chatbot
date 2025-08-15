import os
import io
import html
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import docx

# langchain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from supabase.client import Client, create_client
from langchain_core.messages import HumanMessage, AIMessage

# ---- Load environment variables ----
load_dotenv()

# ---- Initialize Supabase ----
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# ---- Cached initializations ----
@st.cache_resource
def init_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )
    return vector_store

@st.cache_resource
def init_agent():
    vector_store = init_vectorstore()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # Wichtig: Wir geben Content (für das LLM) und Artefakt (für UI-Quellenanzeige) zurück
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """
        Retrieve information related to a query from the vector store.
        Returns HTML-ready content (mit Inline-Quelle) und ein Artefakt mit Roh-Quellen.
        """
        retrieved_docs = vector_store.similarity_search(query, k=4)

        # Content an das LLM (kann ignoriert/umformuliert werden – deshalb UI-seitige Extraktion!)
        serialized_content = "\n\n".join(
            f"Content: {doc.page_content}\n\n"
            f"<span style='color:gray; font-size:small; cursor:help;' "
            f"title='{html.escape(str(doc.metadata.get('source', 'Unbekannte Quelle')), quote=True)}'>Quelle</span>"
            for doc in retrieved_docs
        )

        # Artefakt: möglichst roh & stabil
        artifact = []
        for doc in retrieved_docs:
            artifact.append({
                "source": str(doc.metadata.get("source", "Unbekannte Quelle")),
                "page": doc.metadata.get("page", None),
                "score": getattr(doc, "score", None),
            })

        return serialized_content, artifact

    tools = [retrieve]
    agent = create_tool_calling_agent(llm, tools, prompt)

    # WICHTIG: intermediate_steps zurückgeben, damit wir die Artefakte auslesen können
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    return agent_executor

# ---- Initialize once ----
agent_executor = init_agent()

# ---- Helper: Chat-Titel ----
def generate_chat_title(first_prompt: str) -> str:
    # Kleiner, robuster Titelgenerator
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    title = llm.predict(
        f"Erzeuge einen sehr kurzen Titel für einen Chat basierend auf: '{first_prompt}'. "
        f"Keine Anführungszeichen, max. 6 Wörter."
    ).strip()
    return title or f"Chat {len(st.session_state.chats) + 1}"

# ---- Helper: Quellen aus intermediate_steps extrahieren ----
def build_sources_html_from_steps(result_dict) -> str:
    """
    Sucht in den intermediate_steps nach Tool-Outputs mit Artefakten.
    Erwartet eine Liste von Dicts [{'source': ...}, ...] oder Ähnliches.
    Baut die Tooltip-HTML zusammen. Dedupliziert Quellen.
    """
    steps = result_dict.get("intermediate_steps", []) or []
    found_sources = []

    for step in steps:
        # step kann Tuple (action, observation) oder Dict sein; wir handeln defensiv
        action = None
        observation = None
        if isinstance(step, (list, tuple)) and len(step) == 2:
            action, observation = step
        else:
            observation = step

        # Versuche, ein "artifact" Feld zu bekommen (ToolMessage-ähnlich oder Dict)
        artifact = None
        # 1) Wenn observation ein Objekt mit Attribut 'artifact' ist
        if hasattr(observation, "artifact"):
            artifact = getattr(observation, "artifact", None)
        # 2) Wenn observation ein Dict ist
        if artifact is None and isinstance(observation, dict):
            artifact = observation.get("artifact") or observation.get("tool_output")

        # 3) In manchen LangChain-Versionen steckt das Artefakt direkt in observation
        if artifact is None and isinstance(observation, (list, tuple)):
            # vielleicht schon die Liste selbst
            artifact = observation

        if not artifact:
            continue

        # Normalisiere zu Liste von Quellen-Strings
        if isinstance(artifact, dict):
            artifact = [artifact]
        if not isinstance(artifact, (list, tuple)):
            artifact = [artifact]

        for item in artifact:
            src = None
            if isinstance(item, dict):
                src = item.get("source")
            else:
                # fallbacks
                src = str(item)
            if src:
                found_sources.append(src)

    # Deduplizieren, Reihenfolge beibehalten
    unique_sources = list(dict.fromkeys(found_sources))

    if not unique_sources:
        return ""

    # Baue HTML (Tooltip), sicher escapen
    html_spans = []
    for src in unique_sources:
        title = html.escape(str(src), quote=True)
        html_spans.append(
            f"<span style='color:gray; font-size:small; cursor:help;' title='{title}'>Quelle</span>"
        )
    return "<br>".join(html_spans)

# ---- Streamlit UI ----
st.set_page_config(page_title="Schnoor - Agentic RAG Chatbot", page_icon="🤖")
st.title("🤖 Schnoor´s Chatbot")

# ---- Session State ----
if "chats" not in st.session_state:
    st.session_state.chats = {"Neuer Chat": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Neuer Chat"

# ---- Sidebar ----
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

# ---- Chatverlauf anzeigen ----
for message in st.session_state.chats[st.session_state.current_chat]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content, unsafe_allow_html=True)

# ---- User Input ----
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
        result = agent_executor.invoke({
            "input": user_question,
            "chat_history": st.session_state.chats[current]
        })

    ai_message = result["output"]
    # NEU: Quellen sauber aus den intermediate_steps extrahieren
    quelle_html = build_sources_html_from_steps(result)

    with st.chat_message("assistant"):
        if quelle_html:
            st.markdown(f"{ai_message}<br><br>{quelle_html}", unsafe_allow_html=True)
        else:
            st.markdown(ai_message, unsafe_allow_html=True)

    st.session_state.chats[current].append(AIMessage(ai_message))
