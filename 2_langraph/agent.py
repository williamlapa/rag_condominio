from langgraph.graph import StateGraph, END
from typing import Dict
from qa_session import QASession
from graph_nodes import AgentState, GraphNodes
from llm_manager import LLMManager
from document_loader import DocumentLoader
from datetime import datetime
from IPython.display import display, Markdown
from config import DOCS_DIR, CACHE_DIR

class SolarCondominiumQA:
    """Agente Q&A especializado para condomínio Solar com OCR e cache"""
        
    
    def __init__(self, docs_directory: str = None, provider: str = "openai"):
        # Garantir que docs_directory tem um valor padrão
        if docs_directory is None:
            from config import DOCS_DIR
            docs_directory = DOCS_DIR
        
        # Inicializar componentes
        self.llm_manager = LLMManager(provider)
        self.document_loader = DocumentLoader(docs_directory)
        self.graph_nodes = GraphNodes(self.llm_manager, self.document_loader)
        
        # Configurar sessão
        self.session = QASession(
            conversation_history=[],
            retriever=None,
            retriever_initialized=False,
            session_id=""
        )
        
        self._build_graph()
    
    def _build_graph(self):
        """Constrói o grafo LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Adicionar nós
        workflow.add_node("decide_initial_step", lambda state: state)
        workflow.add_node("load_docs", self.graph_nodes.load_documents_node)
        workflow.add_node("retrieve_docs", self.graph_nodes.retrieve_documents_node)
        workflow.add_node("generate_answer", self.graph_nodes.generate_answer_node)
        
        # Configurar fluxo
        workflow.set_entry_point("decide_initial_step")
        
        workflow.add_conditional_edges(
            "decide_initial_step",
            self.graph_nodes.decide_next_step,
            {
                "load_docs": "load_docs",
                "retrieve_docs": "retrieve_docs"
            }
        )
        
        workflow.add_edge("load_docs", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        self.app = workflow.compile()
    
    def ask_question(self, question: str, show_process: bool = False) -> str:
        """Faz uma pergunta ao agente"""
        print(f"❓ {question}")
        
        if show_process:
            print("🔄 Processando...")
        
        # Preparar estado inicial
        initial_state = {
            "question": question,
            "documents": [],
            "answer": "",
            "retriever": self.session.retriever,
            "retriever_initialized": self.session.retriever_initialized,
            "conversation_history": self.session.conversation_history,
            "session_context": self.session
        }
        
        # Executar grafo
        current_result = None
        for step in self.app.stream(initial_state):
            if show_process:
                node_name = list(step.keys())[0] if step else "unknown"
                print(f"  🔸 {node_name}")
            current_result = step
        
        # Processar resultado
        if current_result:
            final_state = list(current_result.values())[0]
            answer = final_state.get("answer", "❌ Erro ao processar pergunta")
            
            # Atualizar histórico
            qa_pair = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat()
            }
            self.session.conversation_history.append(qa_pair)
            
            return answer
        
        return "❌ Erro no processamento"
    
    def ask_and_display(self, question: str, show_process: bool = False):
        """Faz pergunta e exibe resposta formatada"""
        answer = self.ask_question(question, show_process)
        print(f"\n💡 **Resposta:**")
        display(Markdown(answer))
        return answer
    
    def show_conversation_history(self, limit: int = 5):
        """Mostra histórico de conversas"""
        print(f"\n📋 **Histórico de Conversas** (últimas {limit}):")
        print("-" * 60)
        
        recent = self.session.conversation_history[-limit:]
        
        for i, qa in enumerate(recent, 1):
            timestamp = qa.get("timestamp", "N/A")
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            
            print(f"\n**{i}.** 🕒 {timestamp}")
            print(f"❓ {question}")
            print(f"💡 {answer[:100]}{'...' if len(answer) > 100 else ''}")
    
    def get_session_info(self):
        """Informações da sessão"""
        return {
            "session_id": self.session.session_id,
            "total_questions": len(self.session.conversation_history),
            "retriever_initialized": self.session.retriever_initialized,
            "documents_loaded": self.session.documents_loaded,
            "docs_directory": self.document_loader.docs_directory,
            "cache_directory": self.document_loader.cache_dir
        }