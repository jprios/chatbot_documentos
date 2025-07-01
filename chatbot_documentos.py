# chatbot_documentos.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
import time
import httpx  # adicione no topo do arquivo, se ainda não estiver



# =============================
# Configuração
# =============================
def carregar_credenciais(dotenv_path: str = ".env"):
    """
    Carrega as credenciais do arquivo .env e retorna como dicionário.

    Args:
        dotenv_path (str): Caminho para o arquivo .env (padrão: .env na raiz).

    Returns:
        dict: Contém as chaves 'LANGSMITH_API_KEY' e 'MISTRAL_API_KEY'
    """
    load_dotenv(dotenv_path)

    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    if not langsmith_key or not mistral_key:
        raise ValueError("Credenciais não encontradas no arquivo .env.")

    return {
        "LANGSMITH_API_KEY": langsmith_key,
        "MISTRAL_API_KEY": mistral_key
    }


# =============================
# Indexação do PDF
# =============================
def indexar_documento(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    partes = splitter.split_documents(documentos)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    store = InMemoryVectorStore(embeddings)
    store.add_documents(partes)
    return store


# =============================
# Criação da ferramenta
# =============================
def criar_ferramenta_retrieve(store):
    @tool(response_format="content")
    def retrieve(query: str):
        """Recupera os trechos do documento mais relevantes com base na pergunta."""
        docs = store.similarity_search(query, k=5)
        return "\n\n".join(doc.page_content for doc in docs)

    return retrieve  # <--- ESSENCIAL


# =============================
# Execução do agente
# =============================
def responder_pergunta(store, pergunta_usuario: str) -> str:
    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
    retrieve_tool = criar_ferramenta_retrieve(store)

    def query_or_respond(state: MessagesState):
        resposta = llm.bind_tools([retrieve_tool]).invoke(state["messages"])
        return {"messages": [resposta]}

    def generate(state: MessagesState):
        tool_msgs = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
        tool_msgs = tool_msgs[::-1]
        contexto = tool_msgs[-1].content if tool_msgs else ""

        mensagem_sistema = SystemMessage(
            content=(
                "Você é um assistente de perguntas e respostas. Use o contexto a seguir "
                "para responder com no máximo 3 frases. Seja conciso. "
                "Se não souber, diga que não sabe.\n\n" + contexto
            )
        )

        mensagens_conversacao = [
            m for m in state["messages"]
            if m.type in ("human", "system") or (m.type == "ai" and not m.tool_calls)
        ]

        prompt = [mensagem_sistema] + mensagens_conversacao
        resposta = llm.invoke(prompt)
        return {"messages": [resposta]}

    builder = StateGraph(MessagesState)
    builder.add_node("query_or_respond", query_or_respond)
    builder.add_node("tools", ToolNode([retrieve_tool]))
    builder.add_node("generate", generate)

    builder.set_entry_point("query_or_respond")
    builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    builder.add_edge("tools", "generate")
    builder.add_edge("generate", END)

    memory = MemorySaver()
    agente = create_react_agent(llm, [retrieve_tool], checkpointer=memory)

    input_data = {"messages": [{"role": "user", "content": pergunta_usuario}]}
    config = {"configurable": {"thread_id": "chat-tese"}}

    for attempt in range(10):
        try:
            resultado = agente.invoke(input_data, config=config)
            return resultado["messages"][-1].content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"Tentativa {attempt+1}: limite excedido (429). Aguardando 5 segundos...")
                time.sleep(5)
                continue
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Erro ao processar a pergunta: {e}")
