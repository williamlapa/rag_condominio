import tiktoken

def count_tokens(text: str, model_name: str = "deepseek-chat") -> int:
    """Conta o número de tokens usando o codificador do modelo"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except:
        # Fallback simples se não conseguir contar tokens
        return len(text.split()) // 3  # Aproximação grosseira