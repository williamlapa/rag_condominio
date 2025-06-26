## Exemplo de LangGraph para Q&A com Documentos de Condomínio Residencial

Este exemplo ilustra como usar **LangGraph** para construir um sistema de perguntas e respostas (Q&A) que pode ler de múltiplos arquivos PDF (atas, contratos, comunicados, etc.) de um condomínio em um diretório. O agente será capaz de:

* Processar documentos PDF relevantes para condomínios.
* Criar um índice de busca a partir desses documentos.
* Responder a perguntas consultando esse índice, fornecendo informações específicas do condomínio.

Este é um exemplo mais complexo, pois envolve várias etapas e ferramentas, ideal para demonstrar o poder do LangGraph na orquestração de um fluxo de trabalho.

### Pré-requisitos

Para que este código funcione, você precisará instalar as seguintes bibliotecas:

* `langchain-openai`: Para interagir com os modelos da OpenAI.
* `langchain-community`: Para carregadores de documentos e vetorizadores.
* `langgraph`: O framework de grafo.
* `pypdf`: Para ler arquivos PDF.
* `chromadb`: Um banco de dados vetorial leve e embutido.

**Bash**

```
pip install langchain-openai langchain-community langgraph pypdf chromadb
```

Além disso, certifique-se de que sua **chave de API da OpenAI** esteja configurada como uma variável de ambiente (`OPENAI_API_KEY`) ou diretamente no código.

### Estrutura do Diretório e Documentos

Crie um diretório chamado `docs_condominio` no mesmo local do seu script Python e coloque alguns arquivos PDF dentro dele que simulem documentos de um condomínio (atas de reunião, regulamentos, comunicados, etc.). Por exemplo:

/solar_trindade_qna
└── docs_condominio/
    └── assembleias
    └── contratos
    └── convencao  
├── __init__.py
├── config.py          # Configurações globais e constantes
├── document_loader.py # Carregamento e processamento de documentos
├── llm_manager.py     # Gerenciamento dos modelos LLM
├── qa_session.py      # Classe QASession e estado do agente
├── graph_nodes.py     # Nós do grafo LangGraph
├── agent.py           # Classe principal SolarCondominiumQA
├── main.py            # Ponto de entrada para execução
├── main_antiga.py     # versão antiga do main
└── utils.py           # Funções utilitárias

### Didática Detalhada do Código:

1. **`AgentState` (A Memória do Agente Condominial):**
   * Definimos uma `TypedDict` para o estado. É a "memória" que o agente transporta entre os nós.
   * `question`: A pergunta atual do condômino.
   * `documents`: Os pedaços de PDFs (atas, regulamentos, etc.) carregados ou recuperados.
   * `answer`: A resposta final gerada para o condômino.
   * `retriever`: O objeto de busca (**ChromaDB** nesse caso) que contém os embeddings dos documentos do condomínio.
   * `retriever_initialized`: Uma flag booleana crucial que indica se os PDFs já foram carregados e o `retriever` configurado. Isso evita recarregar tudo a cada nova pergunta.
2. **Nós do Grafo (Etapas do Processo de Q&A):**
   * **`load_documents_node`** :
   * Este nó é o responsável por lidar com a base de conhecimento.
   * Usa `PyPDFDirectoryLoader` para **ler todos os PDFs** do seu diretório `docs_condominio`.
   * Em seguida, `RecursiveCharacterTextSplitter` quebra esses documentos grandes em **partes menores** (`chunks`). Isso é vital para que o LLM possa processar o contexto de forma eficaz.
   * Cria um `Chroma` (nosso banco de dados vetorial) e popula-o com os embeddings (representações numéricas) desses chunks, gerados por `OpenAIEmbeddings`.
   * Por fim, transforma o `Chroma` em um `retriever`, que é a "ferramenta" que sabe buscar os chunks mais relevantes para uma dada pergunta.
   * **Atualiza o estado** com o `retriever` e a flag `retriever_initialized` como `True`.
   * **`retrieve_documents_node`** :
   * Este nó recebe a `question` do estado e usa o `retriever` (criado no nó anterior) para encontrar os **documentos mais relevantes** do seu banco de dados vetorial.
   * Ele é o **componente de busca de informações** específico para o contexto do condomínio.
   * **`generate_answer_node`** :
   * Recebe a `question` e os `documents` relevantes (recuperados pelo nó anterior).
   * Define um `ChatPromptTemplate` que instrui o LLM a atuar como um "assistente de Q&A para condomínios" e a usar o contexto fornecido.
   * Usa `create_stuff_documents_chain` para combinar o LLM com o prompt e os documentos, gerando a resposta final.
   * **Atualiza o estado** com a `answer` gerada.
3. **Arestas e Lógica Condicional (`decide_next_step`):**
   * A função `decide_next_step` é uma  **aresta condicional** , o "ponto de decisão" do nosso agente.
   * Ela verifica a flag `retriever_initialized` no estado:
     * **`False` (primeira pergunta):** O fluxo vai para `load_documents_node`. Isso garante que os documentos sejam carregados e o índice construído apenas uma vez, na primeira pergunta.
     * **`True` (perguntas subsequentes):** O fluxo pula `load_documents_node` e vai direto para `retrieve_documents_node`, economizando tempo e recursos, pois o índice já está pronto.
   * `add_edge` define transições diretas, e `END` marca o fim do processo do grafo.
4. **`app.stream()` (Visualizando o Fluxo):**
   * Usamos `app.stream()` para executar o grafo. Ele retorna o estado do agente a cada transição de nó, o que é incrivelmente útil para **visualizar e depurar** o caminho que o agente está tomando, além de permitir o encadeamento de perguntas.

Este exemplo demonstra um sistema de Q&A robusto para um condomínio. A modularidade do LangGraph permite que você adicione facilmente mais funcionalidades, como verificar a relevância da resposta, pedir mais informações se a resposta for insuficiente, ou até mesmo acionar outras ferramentas para ações específicas do condomínio (como agendar uma manutenção, se fosse um sistema transacional).
