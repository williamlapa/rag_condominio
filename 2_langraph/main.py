from agent import SolarCondominiumQA
from IPython.display import display, Markdown
from datetime import datetime
import os

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

class ConsolidatedReportManager:
    """Gerencia relatório consolidado de todos os modelos testados"""
    
    def __init__(self):
        self.consolidated_data = {}
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def add_model_responses(self, provider, model_name, qa_agent):
        """Adiciona respostas de um modelo ao relatório consolidado"""
        if not qa_agent.session.conversation_history:
            return
            
        self.consolidated_data[provider] = {
            'model_name': model_name,
            'responses': qa_agent.session.conversation_history
        }
    
    def export_consolidated_report(self, filename=None):
        """Exporta relatório consolidado comparando todos os modelos"""
        if not self.consolidated_data:
            print("⚠️ Nenhum dado consolidado para exportar")
            return
        
        if filename is None:
            filename = f"relatorio_consolidado_modelos_{self.session_timestamp}.md"
        
        # Cabeçalho do relatório
        markdown_content = f"""# Relatório Consolidado - Condomínio Solar Trindade

**Data da Avaliação**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  
**Modelos Testados**: {len(self.consolidated_data)}  
**Total de Perguntas**: {len(list(self.consolidated_data.values())[0]['responses']) if self.consolidated_data else 0}

## Modelos Avaliados

"""
        
        # Lista dos modelos testados
        for provider, data in self.consolidated_data.items():
            markdown_content += f"- **{provider.upper()}**: {data['model_name']}\n"
        
        markdown_content += "\n---\n\n"
        
        # Para cada pergunta, mostra respostas de todos os modelos
        if self.consolidated_data:
            total_questions = len(list(self.consolidated_data.values())[0]['responses'])
            
            for q_idx in range(total_questions):
                # Pega a pergunta (assumindo que é a mesma para todos os modelos)
                question = list(self.consolidated_data.values())[0]['responses'][q_idx]['question']
                
                markdown_content += f"## Pergunta {q_idx + 1}\n\n**{question}**\n\n"
                
                # Respostas de cada modelo
                for provider, data in self.consolidated_data.items():
                    if q_idx < len(data['responses']):
                        answer = data['responses'][q_idx]['answer']
                        # Resumir resposta se muito longa (mantém primeiros 500 chars + indicação)
                        if len(answer) > 800:
                            answer = answer[:500] + "\n\n*[Resposta resumida - versão completa disponível nos logs individuais]*"
                        
                        markdown_content += f"### {provider.upper()} ({data['model_name']})\n\n{answer}\n\n"
                
                markdown_content += "---\n\n"
        
        # Seção de observações
        markdown_content += """## Observações Metodológicas

- **Objetivo**: Comparar respostas de diferentes modelos LLM nas mesmas perguntas sobre documentos do condomínio
- **Critérios de Avaliação**: Precisão, relevância, completude e objetividade das respostas
- **Limitações**: Respostas muito longas foram resumidas para facilitar comparação
- **Fonte dos Dados**: Base de conhecimento do Condomínio Solar Trindade (RAG)

---

*Relatório gerado automaticamente pelo Sistema de Q&A do Condomínio Solar Trindade*
"""
        
        # Escreve no arquivo
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✅ Relatório consolidado exportado para '{filename}'")
            return filename
        except Exception as e:
            print(f"❌ Erro ao exportar relatório consolidado: {str(e)}")
            return None

def export_responses_to_markdown(qa_agent, filename="respostas_condominio.md"):
    """Exporta respostas individuais (mantida para compatibilidade)"""
    if not qa_agent.session.conversation_history:
        print("⚠️ Nenhuma resposta no histórico para exportar")
        return
    
    # Obter nome do modelo de forma segura
    model_name = get_model_name(qa_agent.llm_manager.llm)
    
    # Cabeçalho do arquivo
    markdown_content = f"""# Relatório Individual - Condomínio Solar Trindade\n\n
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

def run_solar_qa_demo(provider: str = "openai", report_manager=None):
    """Executa demonstração do agente Solar Q&A com exportação"""
    print(f"\n🏢 === TESTANDO MODELO: {provider.upper()} ===")
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
        # 📝 Atas e Reuniões
        "Quando ocorreu a última assembleia ou reunião registrada do condomínio?",
        "Liste as datas das cinco assembleias ou reuniões mais recentes, em ordem cronológica.",
        "Quais foram os tópicos principais discutidos na assembleia mais recente do condomínio?",
        "Quem são os atuais membros do conselho fiscal, conforme os documentos mais recentes?",
        "De acordo com a última ata de assembleia, qual é o valor atual da taxa condominial definida para os condôminos?",

        # 📄 Contratos
        "Quais contratos ativos o condomínio possui atualmente e com quais empresas foram firmados?",
        "Qual é o valor mensal acordado no contrato de prestação de serviços manutenção de elevadores?",
        "Há cláusulas com penalidades previstas para rescisão antecipada de algum contrato? Se sim, quais?",
        "Quando foi firmado o contrato mais recente e qual é o prazo de vigência previsto?",
        "Existe algum contrato relacionado à manutenção predial, elevadores ou segurança eletrônica? Qual o conteúdo principal?"
    ]

    for i, question in enumerate(sample_questions, 1):
        print(f"\n🔸 **Pergunta {i}/{len(sample_questions)}**")
        qa_agent.ask_and_display(question, show_process=(i == 1))
        if i < len(sample_questions):
            print("\n" + "-"*40)
    
    # Adiciona ao relatório consolidado se o manager foi fornecido
    if report_manager:
        report_manager.add_model_responses(provider, model_name, qa_agent)
    
    # Exporta as respostas individuais (opcional)
    # export_filename = f"respostas_{provider}_{qa_agent.session.session_id.split('_')[1]}.md"
    # export_responses_to_markdown(qa_agent, export_filename)
    
    return qa_agent

if __name__ == "__main__":
    
    # Inicializa o gerenciador de relatório consolidado
    report_manager = ConsolidatedReportManager()
    
    # Escolha dos modelos
    providers = [
        "openai", 
        "gemini", 
        # "claude",        
        "deepseek"
    ]

    print("🚀 INICIANDO AVALIAÇÃO COMPARATIVA DE MODELOS")
    print("="*80)

    # Testa cada modelo
    for provider in providers:
        try:
            run_solar_qa_demo(provider, report_manager)
        except Exception as e:
            print(f"❌ Erro ao testar {provider}: {str(e)}")
            continue
    
    # Exporta o relatório consolidado final
    print("\n" + "="*80)
    print("📊 GERANDO RELATÓRIO CONSOLIDADO...")
    consolidated_filename = report_manager.export_consolidated_report()
    
    if consolidated_filename:
        print(f"🎉 Avaliação completa! Relatório disponível em: {consolidated_filename}")
    else:
        print("⚠️ Não foi possível gerar o relatório consolidado")