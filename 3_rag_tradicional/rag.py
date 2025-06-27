# RAG para Documentos de Condomínio - Versão Otimizada

# RAG para Documentos de Condomínio - Versão Otimizada

import os
import glob
from dotenv import load_dotenv
import gradio as gr

# Imports LangChain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate

# ============================================================================
# 1. CONFIGURAÇÃO INICIAL
# ============================================================================

# Configuração de modelos e diretórios
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"
# 1. Primeiro, verifique se o caminho está correto


# Carregamento de variáveis de ambiente
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# ============================================================================
# 2. CARREGAMENTO E PROCESSAMENTO DE DOCUMENTOS
# ============================================================================

def load_documents():
    """Carrega documentos de forma organizada e com metadados apropriados"""
    
    # Tenta diferentes caminhos possíveis
    possible_paths = [
        os.path.abspath(os.path.join(os.getcwd(), "../0_base_conhecimento/processed_docs_cache")),
        os.path.abspath(os.path.join(os.getcwd(), "0_base_conhecimento/processed_docs_cache")),
        os.path.abspath(os.path.join(os.getcwd(), "./processed_docs_cache")),
        os.path.abspath(os.path.join(os.getcwd(), "../processed_docs_cache"))
    ]
    
    CACHE_DIR = None
    for path in possible_paths:
        if os.path.exists(path):
            CACHE_DIR = path
            print(f"✅ Diretório encontrado: {CACHE_DIR}")
            break
    
    if CACHE_DIR is None:
        print("❌ Diretório não encontrado em nenhum dos caminhos:")
        for path in possible_paths:
            print(f"   - {path}")
        return []

    # BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "../0_base_conhecimento"))
    # CACHE_DIR = os.path.join(BASE_DIR, "processed_docs_cache")
    
    print(f"📁 Conteúdo do diretório: {os.listdir(CACHE_DIR)}")
    
    documents = []
    text_loader_kwargs = {'encoding': 'utf-8'}
    
    # Se não há subdiretórios, carrega diretamente do diretório principal
    subdirs = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
    
    if not subdirs:
        print("📄 Não há subdiretórios, carregando do diretório principal...")
        subdirs = ["."]  # Carrega do próprio diretório
        base_path = CACHE_DIR
    else:
        base_path = CACHE_DIR
    
    # Busca por subdiretórios (tipos de documento)
    for subfolder in subdirs:
        if subfolder == ".":
            subfolder_path = base_path
            doc_type = "documentos"
        else:
            subfolder_path = os.path.join(base_path, subfolder)
            doc_type = subfolder
            
        print(f"📄 Processando: {doc_type} em {subfolder_path}")
        
        # Lista arquivos no diretório
        all_files = []
        for ext in ['*.txt', '*.docx']:
            pattern = os.path.join(subfolder_path, ext)
            files = glob.glob(pattern)
            all_files.extend(files)
        
        print(f"   Arquivos encontrados: {len(all_files)}")
        for file in all_files[:5]:  # Mostra até 5 arquivos
            print(f"   - {os.path.basename(file)}")
        
        # Carrega arquivos .txt
        txt_docs = []
        for txt_file in glob.glob(os.path.join(subfolder_path, "*.txt")):
            try:
                loader = TextLoader(txt_file, encoding='utf-8')
                docs = loader.load()
                txt_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ Erro ao carregar {txt_file}: {e}")
        
        # Carrega arquivos .docx
        docx_docs = []
        for docx_file in glob.glob(os.path.join(subfolder_path, "*.docx")):
            try:
                loader = UnstructuredFileLoader(docx_file)
                docs = loader.load()
                docx_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ Erro ao carregar {docx_file}: {e}")
        
        # Processa e adiciona metadados
        for doc in txt_docs + docx_docs:
            # Limpa e padroniza o conteúdo
            content = str(doc.page_content).strip()
            if len(content) < 50:  # Ignora documentos muito pequenos
                print(f"   ⚠️ Documento muito pequeno ignorado: {len(content)} chars")
                continue
                
            # Cria documento com metadados enriquecidos
            new_doc = Document(
                page_content=content,
                metadata={
                    "doc_type": doc_type,
                    "source": doc.metadata.get("source", "unknown"),
                    "filename": os.path.basename(doc.metadata.get("source", "unknown")),
                    "content_length": len(content)
                }
            )
            documents.append(new_doc)
        
        loaded_count = len(txt_docs + docx_docs)
        valid_count = sum(1 for doc in txt_docs + docx_docs if len(str(doc.page_content).strip()) >= 50)
        print(f"✅ {doc_type}: {loaded_count} arquivos carregados, {valid_count} válidos")
    
    print(f"📊 Total de documentos válidos: {len(documents)}")
    
    if documents:
        # Mostra estatísticas dos documentos
        doc_types = {}
        total_chars = 0
        for doc in documents:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            total_chars += doc.metadata.get('content_length', 0)
        
        print(f"📈 Estatísticas:")
        print(f"   - Total de caracteres: {total_chars:,}")
        print(f"   - Média por documento: {total_chars//len(documents):,} chars")
        print(f"   - Tipos de documento: {dict(doc_types)}")
    
    return documents

def create_chunks(documents):
    """Cria chunks otimizados para documentos de condomínio"""
    
    if not documents:
        print("❌ Nenhum documento para processar")
        return []
    
    # Splitter otimizado para documentos legais/administrativos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Menor para melhor precisão
        chunk_overlap=150,  # Overlap maior para manter contexto
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Adiciona informações extras aos metadados dos chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    print(f"🔗 Total de chunks criados: {len(chunks)}")
    print(f"📋 Tipos de documento encontrados: {set(doc.metadata['doc_type'] for doc in documents)}")
    
    return chunks

# ============================================================================
# 3. CRIAÇÃO DO VECTOR STORE
# ============================================================================

def create_vectorstore(chunks, force_recreate=False):
    """Cria vectorstore otimizado com opção de incremental"""
    
    if not chunks:
        print("❌ Nenhum chunk fornecido para o vectorstore")
        return None
    
    try:
        # Embeddings com modelo mais recente
        print("🔧 Criando embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Verifica se já existe um vectorstore
        vectorstore_exists = os.path.exists(DB_NAME)
        
        if vectorstore_exists and not force_recreate:
            print(f"📂 Vectorstore existente encontrado: {DB_NAME}")
            
            # Carrega vectorstore existente
            vectorstore = Chroma(
                persist_directory=DB_NAME,
                embedding_function=embeddings,
                collection_name="condo_docs"
            )
            
            existing_count = vectorstore._collection.count()
            print(f"📊 Documentos existentes: {existing_count}")
            
            # Verifica quais documentos são novos
            existing_sources = set()
            try:
                # Pega metadados existentes para comparar
                existing_data = vectorstore._collection.get(include=["metadatas"])
                existing_sources = {
                    meta.get("source", "") + "_" + meta.get("filename", "") 
                    for meta in existing_data["metadatas"]
                }
            except:
                print("⚠️ Não foi possível verificar documentos existentes")
            
            # Filtra apenas chunks novos
            new_chunks = []
            for chunk in chunks:
                chunk_id = chunk.metadata.get("source", "") + "_" + chunk.metadata.get("filename", "")
                if chunk_id not in existing_sources:
                    new_chunks.append(chunk)
            
            if new_chunks:
                print(f"➕ Adicionando {len(new_chunks)} novos chunks...")
                vectorstore.add_documents(new_chunks)
                new_count = vectorstore._collection.count()
                print(f"✅ Total após adição: {new_count} documentos (+{new_count - existing_count})")
            else:
                print("ℹ️  Nenhum documento novo encontrado")
            
        else:
            if vectorstore_exists:
                # Remove database anterior se forçando recriação
                import shutil
                shutil.rmtree(DB_NAME)
                print(f"🗑️ Database anterior removido: {DB_NAME}")
            
            # Testa embeddings com um chunk pequeno primeiro
            print("🧪 Testando embeddings...")
            test_embedding = embeddings.embed_query("teste")
            print(f"✅ Embeddings funcionando - dimensões: {len(test_embedding)}")
            
            # Cria novo vectorstore
            print(f"💾 Criando novo vectorstore com {len(chunks)} chunks...")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_NAME,
                collection_name="condo_docs"
            )
            
            count = vectorstore._collection.count()
            print(f"✅ Vectorstore criado com {count:,} documentos")
        
        # Teste básico de busca
        test_results = vectorstore.similarity_search("assembleia", k=1)
        print(f"🔍 Teste de busca: {len(test_results)} resultados")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ ERRO na criação do vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 4. CONFIGURAÇÃO DO CHAT RAG
# ============================================================================

def setup_rag_chain(vectorstore):
    """Configura a cadeia RAG com prompt customizado"""
    
    # Prompt customizado para documentos de condomínio
    custom_prompt = PromptTemplate(
        template="""Você é um assistente especializado em documentos de condomínio. 
        Use as informações fornecidas para responder de forma precisa e detalhada.

        Contexto dos documentos:
        {context}

        Histórico da conversa:
        {chat_history}

        Pergunta: {question}

        Instruções:
        - Responda baseado APENAS nas informações dos documentos fornecidos
        - Se não encontrar a informação, diga claramente que não está disponível nos documentos
        - Cite o tipo de documento quando relevante (ata, contrato, etc.)
        - Seja preciso com datas, valores e nomes
        - Use formatação clara para listas e informações importantes

        Resposta:""",
        input_variables=["context", "chat_history", "question"]
    )
    
    # LLM
    llm = ChatOpenAI(
        temperature=0.3,  # Baixa temperatura para respostas mais precisas
        model_name=MODEL
    )
    
    # Memória
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    # Retriever com configurações otimizadas
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # Número de chunks a recuperar
    )
    
    # Chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

# ============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ============================================================================

# Carregamento e processamento
print("🚀 Iniciando processamento...")

# Passo 1: Carregar documentos
documents = load_documents()
if not documents:
    print("❌ ERRO: Nenhum documento foi carregado!")
    print("Verifique:")
    print("1. Se o diretório '../0_base_conhecimento/processed_docs_cache' existe")
    print("2. Se há arquivos .txt ou .docx nas subpastas")
    print("3. Se os arquivos têm conteúdo suficiente")
    exit()

# Passo 2: Criar chunks
chunks = create_chunks(documents)
if not chunks:
    print("❌ ERRO: Nenhum chunk foi criado!")
    exit()

# Passo 3: Criar vectorstore (modo incremental por padrão)
print("\n🔧 Criando/Atualizando vectorstore...")
vectorstore = create_vectorstore(chunks, force_recreate=True)  # Mude para True se quiser recriar

if vectorstore is None:
    print("❌ ERRO: Falha na criação do vectorstore!")
    exit()

# Passo 4: Testar vectorstore
try:
    test_query = vectorstore.similarity_search("assembleia", k=1)
    print(f"✅ Teste do vectorstore: {len(test_query)} resultados encontrados")
except Exception as e:
    print(f"❌ ERRO no teste do vectorstore: {e}")
    exit()

# Passo 5: Configuração do RAG
try:
    conversation_chain = setup_rag_chain(vectorstore)
    print("✅ Sistema RAG configurado com sucesso!")
except Exception as e:
    print(f"❌ ERRO na configuração do RAG: {e}")
    exit()

# ============================================================================
# 6. TESTE COM PERGUNTAS EXEMPLO
# ============================================================================

def test_questions():
    """Testa o sistema com perguntas exemplo"""
    
    sample_questions = [
        "Quando ocorreu a última assembleia registrada?",
        "Quais são os contratos ativos do condomínio?",
        "Qual o valor atual da taxa condominial?",
        "Quem são os membros do conselho fiscal?",
        "Existe contrato de manutenção de elevadores?"
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTANDO SISTEMA COM PERGUNTAS EXEMPLO")
    print("="*60)
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n📝 Pergunta {i}: {question}")
        try:
            result = conversation_chain.invoke({"question": question})
            answer = result['answer']
            sources = result.get('source_documents', [])
            
            print(f"💬 Resposta: {answer}")
            
            if sources:
                print(f"📄 Fontes ({len(sources)}): ", end="")
                doc_types = [doc.metadata.get('doc_type', 'unknown') for doc in sources[:3]]
                print(", ".join(set(doc_types)))
            
        except Exception as e:
            print(f"❌ Erro: {e}")
        
        print("-" * 40)

# Executar teste
# test_questions()

# ============================================================================
# 7. INTERFACE GRADIO (OPCIONAL - EXECUTAR APENAS SE TUDO ANTERIOR FUNCIONOU)
# ============================================================================

def chat_function(message, history):
    """Função de chat para Gradio"""
    try:
        result = conversation_chain.invoke({"question": message})
        return result["answer"]
    except Exception as e:
        return f"Erro ao processar pergunta: {str(e)}"

def launch_gradio():
    """Lança interface Gradio apenas se o sistema estiver funcionando"""
    try:
        # Teste rápido do sistema
        test_result = conversation_chain.invoke({"question": "teste"})
        
        print("\n🌐 Sistema funcionando! Iniciando interface Gradio...")
        interface = gr.ChatInterface(
            fn=chat_function,
            type="messages",
            title="🏢 Assistente de Condomínio RAG",
            description="Faça perguntas sobre os documentos do condomínio (atas, contratos, etc.)",
            examples=[
                "Quando foi a última assembleia?",
                "Quais contratos estão vigentes?",
                "Qual o valor da taxa condominial?",
                "Quem são os síndicos atuais?"
            ]
        )
        
        # Lançar interface
        interface.launch(
            inbrowser=True,
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )
        
    except Exception as e:
        print(f"❌ Erro ao lançar Gradio: {e}")
        print("💡 Execute apenas as funções de teste por enquanto")

# ============================================================================
# 8. FUNÇÕES UTILITÁRIAS PARA GERENCIAR DOCUMENTOS
# ============================================================================

def add_new_document(file_path, doc_type="novo_documento"):
    """Adiciona um único documento novo ao vectorstore existente"""
    try:
        # Carrega o documento
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.docx'):
            loader = UnstructuredFileLoader(file_path)
        else:
            print(f"❌ Formato não suportado: {file_path}")
            return False
        
        docs = loader.load()
        
        # Processa o documento
        for doc in docs:
            content = str(doc.page_content).strip()
            if len(content) < 50:
                print(f"⚠️ Documento muito pequeno: {len(content)} chars")
                continue
            
            doc.metadata.update({
                "doc_type": doc_type,
                "filename": os.path.basename(file_path),
                "content_length": len(content)
            })
        
        # Cria chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        
        # Adiciona ao vectorstore existente
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
            collection_name="condo_docs"
        )
        
        old_count = vectorstore._collection.count()
        vectorstore.add_documents(chunks)
        new_count = vectorstore._collection.count()
        
        print(f"✅ Documento adicionado: {file_path}")
        print(f"📊 Chunks criados: {len(chunks)}")
        print(f"📈 Total no banco: {old_count} → {new_count} (+{new_count - old_count})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao adicionar documento: {e}")
        return False

def list_documents_in_vectorstore():
    """Lista todos os documentos no vectorstore"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
            collection_name="condo_docs"
        )
        
        # Pega todos os metadados
        data = vectorstore._collection.get(include=["metadatas"])
        
        # Agrupa por documento
        docs_info = {}
        for meta in data["metadatas"]:
            doc_key = meta.get("filename", "unknown")
            doc_type = meta.get("doc_type", "unknown")
            
            if doc_key not in docs_info:
                docs_info[doc_key] = {
                    "type": doc_type, 
                    "chunks": 0,
                    "source": meta.get("source", "unknown")
                }
            docs_info[doc_key]["chunks"] += 1
        
        print(f"📋 Documentos no vectorstore ({len(docs_info)} arquivos):")
        for filename, info in docs_info.items():
            print(f"  📄 {filename} ({info['type']}) - {info['chunks']} chunks")
        
        total_chunks = sum(info["chunks"] for info in docs_info.values())
        print(f"📊 Total: {total_chunks} chunks")
        
        return docs_info
        
    except Exception as e:
        print(f"❌ Erro ao listar documentos: {e}")
        return {}

def clear_vectorstore():
    """Remove completamente o vectorstore"""
    try:
        if os.path.exists(DB_NAME):
            import shutil
            shutil.rmtree(DB_NAME)
            print(f"🗑️ Vectorstore removido: {DB_NAME}")
            return True
        else:
            print("ℹ️  Vectorstore não existe")
            return False
    except Exception as e:
        print(f"❌ Erro ao remover vectorstore: {e}")
        return False

# Exemplos de uso:
print("\n" + "="*50)
print("🛠️  FUNÇÕES UTILITÁRIAS DISPONÍVEIS:")
print("="*50)
print("1. list_documents_in_vectorstore() - Lista documentos no banco")
print("2. add_new_document('caminho/arquivo.txt', 'tipo') - Adiciona novo documento")
print("3. clear_vectorstore() - Remove todo o banco")
print("4. create_vectorstore(chunks, force_recreate=True) - Recria banco")
print("="*50)

clear_vectorstore()  # Limpa o vectorstore para testes

