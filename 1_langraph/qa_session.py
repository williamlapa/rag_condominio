from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

@dataclass
class QASession:
    """Classe para gerenciar sessão de Q&A com memória"""
    conversation_history: List[Dict[str, str]]
    retriever: Optional[Any]
    retriever_initialized: bool
    session_id: str
    documents_loaded: bool = False
    
    def __post_init__(self):
        if not self.conversation_history:
            self.conversation_history = []
        if not self.session_id:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"