from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import os
from config import SUPPORTED_PROVIDERS, EMBEDDING_MODEL

class LLMManager:
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm = None
        self.embeddings = None
        self._initialize_llm_dynamic(provider)
    
    def _initialize_llm_openai(self):
        """Inicializa LLM e embeddings"""
        print("🔧 Inicializando LLM OpenAI (gpt-4o-mini)...")
        self.llm = ChatOpenAI(model=SUPPORTED_PROVIDERS["openai"], temperature=0)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def _initialize_llm_gemini(self):
        """Versão Gemini"""
        print("🔧 Inicializando LLM Google Gemini...")
        self.llm = ChatGoogleGenerativeAI(
            model=SUPPORTED_PROVIDERS["gemini"],
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def _initialize_llm_claude(self):
        """Versão Claude"""
        print("🔧 Inicializando LLM Anthropic Claude...")
        self.llm = ChatAnthropic(
            model=SUPPORTED_PROVIDERS["claude"],
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
        )
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)    

    def _initialize_llm_deepseek(self):
        """Versão DeepSeek (OpenAI API compatível)"""
        print("🔧 Inicializando LLM DeepSeek (DeepSeek Chat)...")
        self.llm = ChatOpenAI(
            model=SUPPORTED_PROVIDERS["deepseek"],
            temperature=0,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com"
        )
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    def _initialize_llm_dynamic(self, provider: str):
        """Função dinâmica que escolhe o provider"""
        providers = {
            "openai": self._initialize_llm_openai,
            "gemini": self._initialize_llm_gemini,
            "claude": self._initialize_llm_claude,            
            "deepseek": self._initialize_llm_deepseek
        }

        print(f"🔧 Inicializando LLM com: {provider}\n")
        print("="*60)
        
        if provider in providers:
            providers[provider]()
        else:
            available = ", ".join(providers.keys())
            raise ValueError(f"Provider '{provider}' não suportado. Disponíveis: {available}")