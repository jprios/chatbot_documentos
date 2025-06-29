import os
from typing_extensions import TypedDict
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from pyathena import connect
import boto3
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Define um tipo de estado, caso precise expandir depois
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


# Executa uma única vez na inicialização
def set_environment_variables():
    env_path = Path("config_llm.env")
    if not env_path.exists():
        raise FileNotFoundError("Arquivo 'config_llm.env' não encontrado.")
    
    load_dotenv(dotenv_path=env_path)

    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MISTRAL_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING"
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"A variável de ambiente '{var}' não está definida no .env.")


# Conexão com Athena
def get_database():
    bucket = "coget-athena-tmp"
    athena_uri = (
        "awsathena+rest://@athena.{region}.amazonaws.com:443/"
        "{schema}?s3_staging_dir={s3_staging_dir}"
    ).format(
        region="sa-east-1",
        schema="bigdata_ceara",
        s3_staging_dir=f's3://{bucket}/'
    )
    return SQLDatabase.from_uri(
        athena_uri,
        lazy_table_reflection=True,
        include_tables=["gold_programas_sociais"],
        sample_rows_in_table_info=5
    )


# Criação do agente com ferramentas
def build_agent(llm, db):
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_message = """
    Você é um agente projetado para interagir com uma base de dados Athena.
    
    Todas as consultas DEVEM ser feitas única e exclusivamente na tabela chamada `gold_programas_sociais`.

    Ignore qualquer outra tabela. Caso a pergunta não possa ser respondida com os dados da tabela `gold_programas_sociais`, informe isso ao usuário.

    Crie uma consulta SQL sintaticamente correta baseada na pergunta em português,
    execute-a e retorne a resposta também em português. Nunca consulte todas as colunas, apenas quando isso for pedido explicitamente — selecione apenas as relevantes.

    Nunca use comandos de modificação (INSERT, UPDATE, DELETE, DROP).

    Sempre limite os resultados a no máximo 12, a menos que o usuário peça explicitamente mais resultados.

    Verifique a estrutura da tabela antes de gerar sua consulta.
    """.strip()

    return create_react_agent(llm, tools, prompt=system_message)


# Função principal que será chamada pelo frontend
def process_user_question(question: str) -> str:
    if not os.environ.get("MISTRAL_API_KEY"):
        set_environment_variables()

    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
    db = get_database()
    agent_executor = build_agent(llm, db)

    final_response = None

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        messages = step.get("messages", [])
        for msg in messages:
            if msg.type == "ai" and isinstance(msg.content, str):
                # Captura a última mensagem do tipo Ai Message com texto final
                final_response = msg.content.strip()

    return final_response or "Não foi possível obter uma resposta do agente."
