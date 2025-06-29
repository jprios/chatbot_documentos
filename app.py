import streamlit as st
import os
from chatbot_documentos import (
    carregar_credenciais,
    indexar_documento,
    responder_pergunta
)

# =============================
# Configuração do Ambiente
# =============================
try:
    credenciais = carregar_credenciais()
    os.environ["LANGSMITH_API_KEY"] = credenciais["LANGSMITH_API_KEY"]
    os.environ["MISTRAL_API_KEY"] = credenciais["MISTRAL_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"
except ValueError as e:
    st.error(f"Erro ao carregar credenciais: {e}")
    st.stop()

# =============================
# Interface com o Usuário
# =============================
st.set_page_config(layout="centered")
st.title("💬 Chat com Documento PDF")

uploaded_file = st.file_uploader("📄 Faça upload do PDF", type=["pdf"])
pergunta = st.text_input("✍️ Digite sua pergunta:")

if uploaded_file and pergunta:
    with st.spinner("🔍 Processando o documento..."):
        # Salva o arquivo temporariamente
        caminho_pdf = "documento_temp.pdf"
        with open(caminho_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            store = indexar_documento(caminho_pdf)
            resposta = responder_pergunta(store, pergunta)
            st.success("✅ Resposta gerada:")
            st.markdown(resposta)
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar: {e}")
