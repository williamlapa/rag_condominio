import operator
import os
from typing import TypedDict, Annotated, List

# As bibliotecas reais seriam importadas aqui:
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- 1. Definir o Estado do Grafo ---
# Este é o "cérebro" do nosso agente, contendo todas as informações que ele precisa
# para passar entre os diferentes nós do grafo.

class AgentState(TypedDict):
    """
    Representa o estado do nosso grafo de agente de Q&A para condomínios.
    Contém as informações necessárias para as transições entre os nós.
    """
    # Pergunta feita pelo usuário
    question: str
    # Documentos carregados e divididos
    documents: Annotated[List[str], operator.add]
    # Mensagem final ou resposta
    answer: str
    # O retriever para buscar documentos relevantes
    retriever: object # Será um objeto Chroma.as_retriever()
    # Flag para indicar se o retriever foi inicializado
    retriever_initialized: bool

# --- 2. Inicializar Componentes LLM e Embeddings ---
# Certifique-se de que sua chave de API da OpenAI está configurada como variável de ambiente
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# --- 3. Definir as Funções dos Nós (Etapas do Grafo) ---

def load_documents_node(state: AgentState) -> AgentState:
    """
    Nó para carregar documentos PDF de um diretório específico de condomínio.
    Atualiza o estado com os documentos carregados e o retriever.
    """
    print("\n[Nó: Carregar Documentos] Carregando PDFs do diretório 'docs_condominio/'...")
    
    docs_dir = "docs_condominio"
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        print(f"[Nó: Carregar Documentos] Erro: O diretório '{docs_dir}' não existe ou está vazio. Por favor, coloque PDFs lá.")
        # Retorna um estado que indica falha ou que o retriever não foi inicializado
        return {"retriever_initialized": False, "answer": "Não foi possível carregar os documentos para Q&A."}

    loader = PyPDFDirectoryLoader(docs_dir)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    print(f"[Nó: Carregar Documentos] {len(split_docs)} documentos (partes) carregados e processados.")

    # Criação do banco de dados vetorial (Chroma) e do retriever
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    print("[Nó: Carregar Documentos] Retriever inicializado com sucesso.")

    return {
        "documents": split_docs,
        "retriever": retriever,
        "retriever_initialized": True,
        "answer": "" # Reseta a resposta para a nova pergunta
    }

def retrieve_documents_node(state: AgentState) -> AgentState:
    """
    Nó para buscar documentos relevantes usando o retriever.
    """
    print("\n[Nó: Buscar Documentos] Buscando documentos relevantes para a pergunta...")
    question = state["question"]
    retriever = state["retriever"]

    # Certifica-se de que o retriever existe antes de invocar
    if retriever is None:
        print("[Nó: Buscar Documentos] Erro: Retriever não está inicializado. Pulando busca.")
        return {"documents": [], "answer": "Retriever não configurado para busca."}

    documents_for_qa = retriever.invoke(question)

    print(f"[Nó: Buscar Documentos] Encontrados {len(documents_for_qa)} documentos relevantes.")
    return {"documents": documents_for_qa}

def generate_answer_node(state: AgentState) -> AgentState:
    """
    Nó para gerar a resposta usando o LLM e os documentos recuperados.
    """
    print("\n[Nó: Gerar Resposta] Gerando a resposta com base na pergunta e nos documentos...")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("[Nó: Gerar Resposta] Nenhum documento para gerar resposta.")
        return {"answer": "Não foram encontrados documentos relevantes para responder a esta pergunta."}

    # Define o prompt para o LLM
    prompt = ChatPromptTemplate.from_template("""
    Você é um assistente de Q&A para condomínios. Use os seguintes pedaços de contexto
    sobre documentos do condomínio (atas, contratos, comunicados, regulamentos) para responder à pergunta.
    Se você não souber a resposta, diga "Não tenho informações suficientes para responder a isso nos documentos do condomínio.",
    não tente inventar uma resposta.

    Contexto: {context}

    Pergunta: {input}
    """)

    # Cria a cadeia de documentos e a cadeia de recuperação
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Criamos uma cadeia de recuperação simples aqui para usar os documentos já recuperados
    # Note que esta não é a cadeia completa com retriever, mas apenas a parte de combinação de documentos
    answer_chain = document_chain 

    # Invoca a cadeia para obter a resposta
    response = answer_chain.invoke({"input": question, "context": documents})
    answer = response

    print(f"[Nó: Gerar Resposta] Resposta gerada: '{answer[:200]}...'")
    return {"answer": answer}

# --- 4. Definir as Arestas Condicionais (Lógica de Transição) ---

def decide_next_step(state: AgentState) -> str:
    """
    Função condicional para decidir o próximo passo.
    Se o retriever ainda não foi inicializado, primeiro carregamos os documentos.
    Caso contrário, passamos para a busca e geração da resposta.
    """
    if not state.get("retriever_initialized"):
        print("[Decisor de Fluxo] Retriever não inicializado. Indo para 'load_docs'.")
        return "load_docs"
    else:
        print("[Decisor de Fluxo] Retriever já inicializado. Indo para 'retrieve_docs'.")
        return "retrieve_docs"

# --- 5. Construir o Grafo ---

# Criar a instância do grafo com o estado definido
workflow = StateGraph(AgentState)

# Adicionar os nós ao grafo
workflow.add_node("load_docs", load_documents_node)
workflow.add_node("retrieve_docs", retrieve_documents_node)
workflow.add_node("generate_answer", generate_answer_node)

# Definir o ponto de entrada (onde o processo começa)
workflow.set_entry_point("decide_initial_step")

# Adicionar a lógica condicional no ponto de entrada
workflow.add_conditional_edges(
    "decide_initial_step",
    decide_next_step, # A função 'decide_next_step' determina para onde ir
    {
        "load_docs": "load_docs",
        "retrieve_docs": "retrieve_docs"
    }
)

# Adicionar arestas (transições diretas)
workflow.add_edge("load_docs", "retrieve_docs") # Depois de carregar, sempre buscamos
workflow.add_edge("retrieve_docs", "generate_answer") # Depois de buscar, sempre geramos a resposta

# Definir o ponto final do grafo
workflow.add_edge("generate_answer", END) # Após gerar a resposta, o processo termina

# Compilar o grafo
app = workflow.compile()

# --- 6. Executar o Agente Q&A ---

print("--- EXECUTANDO O AGENTE Q&A PARA CONDOMÍNIO COM LANGGRAPH ---")

# Criar o diretório de documentos se não existir
if not os.path.exists("docs_condominio"):
    os.makedirs("docs_condominio")
    print("Diretório 'docs_condominio/' criado. Por favor, coloque seus PDFs de condomínio aqui.")
    print("O script irá falhar se não houver documentos.")


# Para a primeira pergunta, o retriever não está inicializado, então ele carregará os documentos.
# Passamos um estado inicial com retriever_initialized=False
initial_state_q1 = {"question": "Quais foram os principais pontos da última reunião de condomínio sobre áreas comuns?", "retriever_initialized": False}
print(f"\nPrimeira Pergunta: {initial_state_q1['question']}")

# Itera sobre o stream para ver a transição de estados
current_result_q1 = None
for s in app.stream(initial_state_q1):
    print(s)
    current_result_q1 = s # Armazena o último estado

# Pega a resposta final do último estado do stream
final_answer_1 = list(current_result_q1.values())[0]["answer"]
print(f"\n[RESPOSTA FINAL 1]: {final_answer_1}")

print("\n" + "="*80 + "\n")

# Para a segunda pergunta, o retriever já estará inicializado.
# Usamos o estado do resultado anterior para manter o retriever.
# Isso simula a persistência do retriever entre perguntas sem recarregar.
initial_state_q2 = {
    "question": "Qual o horário de uso da piscina conforme o regulamento interno?",
    "retriever_initialized": True,
    "retriever": list(current_result_q1.values())[0]["retriever"] # Reutiliza o retriever do estado anterior
}
print(f"Segunda Pergunta: {initial_state_q2['question']}")

current_result_q2 = None
for s in app.stream(initial_state_q2):
    print(s)
    current_result_q2 = s

final_answer_2 = list(current_result_q2.values())[0]["answer"]
print(f"\n[RESPOSTA FINAL 2]: {final_answer_2}")

print("\n--- FIM DA EXECUÇÃO DO AGENTE ---")