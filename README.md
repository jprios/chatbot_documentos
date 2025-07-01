# Chatbot de Documentos PDF com LangChain e Streamlit

Este projeto permite a interação com documentos PDF por meio de uma interface desenvolvida com Streamlit e um modelo de linguagem da Mistral, utilizando recursos da LangChain para indexação, embeddings e controle de fluxo conversacional.

---

## Funcionalidades

- Upload de arquivos PDF
- Indexação automática do conteúdo com embeddings vetoriais
- Interface de chat com perguntas e respostas baseadas no conteúdo do documento
- Histórico da conversa salvo na sessão
- Uso de agentes e ferramentas com LangGraph para recuperação de trechos relevantes

---

## Estrutura do Projeto

- `app.py`: interface principal construída em Streamlit. Gerencia upload de arquivos, controle de sessão e interface de chat.
- `chatbot_documentos.py`: módulo de backend responsável por carregar credenciais, indexar o documento, criar ferramentas de busca e executar o agente de resposta.

---
