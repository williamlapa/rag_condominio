from agent import SolarCondominiumQA
from IPython.display import display, Markdown

def get_model_name(llm):
    """Função segura para obter o nome do modelo de qualquer provider"""
    try:
        # Para OpenAI e DeepSeek
        if hasattr(llm, 'model_name'):
            return llm.model_name
        # Para Google Gemini
        elif hasattr(llm, 'model'):
            return llm.model
        # Para Claude
        elif hasattr(llm, 'model_id'):
            return llm.model_id
        # Para Groq
        elif hasattr(llm, 'model_name'):
            return llm.model_name
        else:
            return "Modelo desconhecido"
    except:
        return "Modelo não identificado"

def export_responses_to_markdown(qa_agent, filename="respostas_condominio.md"):
    """Exporta todas as respostas do histórico para um arquivo Markdown"""
    if not qa_agent.session.conversation_history:
        print("⚠️ Nenhuma resposta no histórico para exportar")
        return
    
    # Obter nome do modelo de forma segura
    model_name = get_model_name(qa_agent.llm_manager.llm)
    
    # Cabeçalho do arquivo
    markdown_content = f"""# Relatório de Respostas - Condomínio Solar Trindade\n\n
**Modelo utilizado**: {model_name}\n
**Data da sessão**: {qa_agent.session.session_id.split('_')[1]}\n
**Total de perguntas**: {len(qa_agent.session.conversation_history)}\n\n
---\n\n"""
    
    # Adiciona cada Q&A
    for i, qa in enumerate(qa_agent.session.conversation_history, 1):
        markdown_content += f"""## Pergunta {i}\n**{qa['question']}**\n\n
### Resposta\n{qa['answer']}\n\n
---\n\n"""
    
    # Escreve no arquivo
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"✅ Respostas exportadas para '{filename}'")
    except Exception as e:
        print(f"❌ Erro ao exportar: {str(e)}")

def run_solar_qa_demo(provider: str = "openai"):
    """Executa demonstração do agente Solar Q&A com exportação"""
    print("🏢 === AGENTE Q&A CONDOMÍNIO SOLAR TRINDADE ===")
    print("="*60)
    
    qa_agent = SolarCondominiumQA(provider=provider)
    
    # Obter nome do modelo de forma segura
    model_name = get_model_name(qa_agent.llm_manager.llm)
    
    print(f"📊 **Informações da Sessão:**")
    print(f"  • Modelo: {model_name}")
    for key, value in qa_agent.get_session_info().items():
        if key not in ["retriever_initialized", "documents_loaded"]:
            print(f"  • {key}: {value}")
    
    print("\n" + "="*60)
    
    # Perguntas de exemplo
    sample_questions = [
        "Qual a data da reunião ou assembleia do condomínio mais recente?",
        "Quais as datas das cinco reuniões mais recentes?",
        "Quais foram os principais assuntos da reunião mais recente?",
        "Quais os nomes dos membros do conselho fiscal atuais?",
        "Como base na assembleia mais recente, qual o valor da taxa condominial atual?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\n🔸 **Pergunta {i}/{len(sample_questions)}**")
        qa_agent.ask_and_display(question, show_process=(i == 1))
        if i < len(sample_questions):
            print("\n" + "-"*40)
    
    # Exporta as respostas
    export_filename = f"respostas_{provider}_{qa_agent.session.session_id.split('_')[1]}.md"
    export_responses_to_markdown(qa_agent, export_filename)
    
    return qa_agent

def interactive_solar_qa():
    """Modo interativo com opção de exportação"""
    qa_agent = SolarCondominiumQA()
    
    # Obter nome do modelo de forma segura
    model_name = get_model_name(qa_agent.llm_manager.llm)
    
    print("🏢 === MODO INTERATIVO - SOLAR TRINDADE ===")
    print(f"Modelo: {model_name}")
    print("Comandos: 'sair', 'historico', 'info', 'limpar', 'exportar'")
    print("="*60)
    
    while True:
        try:
            question = input("\n❓ Sua pergunta: ").strip()
            
            if question.lower() in ['sair', 'exit', 'quit']:
                # Oferece exportar antes de sair
                if qa_agent.session.conversation_history:
                    export = input("Exportar respostas antes de sair? (s/n): ").lower()
                    if export == 's':
                        export_responses_to_markdown(qa_agent)
                print("👋 Encerrando...")
                break
            elif question.lower() in ['historico', 'history']:
                qa_agent.show_conversation_history()
                continue
            elif question.lower() == 'info':
                print("\n📊 **Informações da Sessão:**")
                print(f"  • Modelo: {model_name}")
                for key, value in qa_agent.get_session_info().items():
                    if key not in ["retriever_initialized", "documents_loaded"]:
                        print(f"  • {key}: {value}")
                continue
            elif question.lower() in ['limpar', 'clear']:
                qa_agent.session.conversation_history = []
                print("🗑️ Histórico limpo")
                continue
            elif question.lower() == 'exportar':
                export_responses_to_markdown(qa_agent)
                continue
            elif not question:
                continue
            
            qa_agent.ask_and_display(question)
            
        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    return qa_agent

if __name__ == "__main__":
    # Escolha o modo de execução:
    # qa_agent = run_solar_qa_demo("claude")     # Demo automática
    # qa_agent = interactive_solar_qa()   # Modo interativo

    providers = [
        "openai", 
        "gemini", 
        # "claude",
        # "groq_llama3", 
        # "groq_gemma", 
        # "groq_mistral", 
        # "deepseek"
    ]

    for provider in providers:
        print(f"\n🔧 Inicializando LLM: {provider}")
        run_solar_qa_demo(provider)