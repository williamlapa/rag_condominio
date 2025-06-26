# graph_nodes.py - Versão melhorada
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from config import DOCS_DIR, CACHE_DIR
import tiktoken

class AgentState(TypedDict):
    """Estado do agente Q&A para condomínios"""
    question: str
    documents: Annotated[List[Document], operator.add]
    answer: str
    retriever: object
    retriever_initialized: bool
    conversation_history: List[Dict[str, str]]
    session_context: Optional[Any]

class GraphNodes:
    def __init__(self, llm_manager, document_loader):
        self.llm = llm_manager.llm
        self.embeddings = llm_manager.embeddings
        self.document_loader = document_loader
        self.persist_dir = os.path.join(CACHE_DIR, "chroma_db")  # Diretório para persistência
        
        # Configurações otimizadas
        self.chunk_size = 1000  # Aumentado para capturar mais contexto
        self.chunk_overlap = 200  # Maior overlap para manter continuidade
        self.search_kwargs = {
            "k": 25,  # Número maior de resultados para melhor recall
            "score_threshold": 0.5,  # Limiar de similaridade
            "filter": None  # Filtros adicionais podem ser adicionados aqui
        }

    def _count_tokens(self, text: str) -> int:
        """Conta tokens com fallback robusto"""
        try:
            encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
            return len(encoding.encode(text))
        except:
            return len(text.split()) // 3

    def _truncate_documents(self, documents: List[Document], max_tokens: int = 60000) -> List[Document]:
        """Trunca documentos com estratégia melhorada"""
        truncated_docs = []
        total_tokens = 0
        
        for doc in sorted(documents, key=lambda x: len(x.page_content), reverse=True):
            content = doc.page_content
            tokens = self._count_tokens(content)
            
            if total_tokens + tokens <= max_tokens:
                truncated_docs.append(doc)
                total_tokens += tokens
            else:
                remaining = max_tokens - total_tokens
                if remaining > 500:  # Só adiciona se valer a pena
                    # Mantém o início do documento (geralmente mais relevante)
                    truncated_content = " ".join(content.split()[:int(remaining*2.5)])
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata=doc.metadata
                    )
                    truncated_docs.append(truncated_doc)
                    break
        
        print(f"ℹ️ Documentos truncados para {total_tokens} tokens (limite: {max_tokens})")
        return truncated_docs

    def load_documents_node(self, state: AgentState) -> AgentState:
        """Nó para carregar documentos com Chroma persistente"""
        if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
            return {
                "retriever_initialized": False, 
                "answer": f"❌ Diretório '{DOCS_DIR}' vazio ou inexistente."
            }

        # Carregar documentos
        docs = self.document_loader.load_documents_with_cache()
        if not docs:
            return {
                "retriever_initialized": False, 
                "answer": "❌ Nenhum documento válido encontrado."
            }

        # Dividir documentos com configurações otimizadas
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens
        )
        split_docs = text_splitter.split_documents(docs)

        # Criar vetorstore persistente com configurações melhoradas
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,  # Persistência em disco
            collection_metadata={"hnsw:space": "cosine"},  # Similaridade cosseno
            collection_name="condominio_docs"  # Nome específico para a coleção
        )

        # Configurar retriever com parâmetros otimizados
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=self.search_kwargs
        )

        return {
            "documents": split_docs,
            "retriever": retriever,
            "retriever_initialized": True,
            "answer": ""
        }

    def retrieve_documents_node(self, state: AgentState) -> AgentState:
        """Nó para buscar documentos com filtro por relevância"""
        question = state["question"]
        retriever = state.get("retriever")
        
        if retriever is None:
            return {
                "documents": [], 
                "answer": "❌ Retriever não configurado."
            }

        try:
            # Busca com filtro de similaridade
            documents_for_qa = retriever.invoke(question)
            
            # Filtro adicional por relevância
            relevant_docs = self._filter_relevant_documents(documents_for_qa, question)
            
            return {"documents": relevant_docs}
        except Exception as e:
            print(f"⚠️ Erro na recuperação: {str(e)}")
            return {"documents": []}

    def _filter_relevant_documents(self, documents: List[Document], question: str) -> List[Document]:
        """Filtro híbrido de relevância"""
        if not documents:
            return []

        # 1. Filtro por similaridade de embeddings
        question_embedding = self.embeddings.embed_query(question)
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        
        similarities = []
        for i, doc in enumerate(documents):
            similarity = np.dot(question_embedding, doc_embeddings[i])
            similarities.append((similarity, doc))
        
        # 2. Ordenar por similaridade
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # 3. Pegar os top N e aplicar limiar
        top_n = 3
        threshold = 0.6  # Limiar mais rigoroso
        filtered = [doc for sim, doc in similarities[:top_n] if sim >= threshold]
        
        return filtered if filtered else documents[:1]  # Fallback para o mais relevante

    def generate_answer_node(self, state: AgentState) -> AgentState:
        """Nó para gerar resposta com prompt otimizado"""
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            return {"answer": "❌ Nenhum documento relevante encontrado."}

        # Trunca documentos com estratégia melhorada
        truncated_docs = self._truncate_documents(documents, max_tokens=40000)  # Limite mais conservador

        # Prompt otimizado para condomínios
        prompt = ChatPromptTemplate.from_template("""
        Você é um especialista em condomínio Solar Trindade. Analise os documentos e responda com:

        REQUISITOS:
        - Precisão: Cite datas, valores e nomes exatamente como nos documentos
        - Contexto: Relacione com outras informações relevantes quando possível
        - Formato:
          • Assunto principal em negrito
          • Detalhes em marcadores
          • Referências entre parênteses

        DOCUMENTOS:
        {context}

        PERGUNTA: 
        {input}

        RESPOSTA (seja direto e específico):
        """)
        
        try:
            chain = create_stuff_documents_chain(self.llm, prompt)
            response = chain.invoke({
                "input": question, 
                "context": truncated_docs
            })
            return {"answer": response}
        except Exception as e:
            return {"answer": f"❌ Erro ao gerar resposta: {str(e)}"}

    def decide_next_step(self, state: AgentState) -> str:
        """Decisor com lógica robusta"""
        if state.get("retriever_initialized", False):
            # Verifica se o diretório de persistência existe
            if os.path.exists(self.persist_dir):
                return "retrieve_docs"
        return "load_docs"