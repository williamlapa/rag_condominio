import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from typing import List
from config import CACHE_DIR, DOCS_DIR, EMBEDDING_MODEL

class DocumentLoader:
    def __init__(self, docs_directory: str = DOCS_DIR, cache_dir: str = CACHE_DIR):
        from config import DOCS_DIR, CACHE_DIR
        
        # Definir valores padrão se None
        self.docs_directory = docs_directory if docs_directory is not None else DOCS_DIR
        self.cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
        
        self._setup_directories()
        
    
    def _setup_directories(self):
        """Cria diretórios necessários"""
        for directory in [self.docs_directory, self.cache_dir]:
            if directory is None:
                raise ValueError("O caminho do diretório não pode ser None")
            
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Diretório '{directory}/' criado.")
    
    def _ocr_pdf_page(self, pdf_doc, page_number):
        """Realiza OCR em uma página do PDF"""
        try:
            page = pdf_doc.load_page(page_number)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang='por')
            return text
        except Exception as e:
            print(f"    ❌ [OCR] Erro na página {page_number+1}: {e}")
            return ""
    
    def load_documents_with_cache(self) -> List[Document]:
        """Carrega documentos com sistema de cache inteligente"""
        documentos_carregados = []
        print(f"📚 Carregando documentos de: {self.docs_directory}")
        
        for root, dirs, files in os.walk(self.docs_directory):
            for file_name in files:
                if file_name.lower().endswith(".pdf"):
                    caminho_arquivo_pdf = os.path.join(root, file_name)
                    cache_file_name = os.path.splitext(file_name)[0] + ".txt"
                    cache_file_path = os.path.join(self.cache_dir, cache_file_name)
                    
                    texto_completo = ""
                    
                    # Tentar carregar do cache
                    if os.path.exists(cache_file_path):
                        try:
                            with open(cache_file_path, "r", encoding="utf-8") as f:
                                texto_completo = f.read()
                            if texto_completo.strip():
                                documentos_carregados.append(
                                    Document(
                                        page_content=texto_completo, 
                                        metadata={"source": caminho_arquivo_pdf, "cached": True}
                                    )
                                )
                                continue
                        except Exception as e:
                            print(f"  ⚠️  Erro no cache para '{file_name}': {e}")
                    
                    # Processar PDF se não estiver no cache
                    try:
                        doc = fitz.open(caminho_arquivo_pdf)
                        print(f"  📄 Processando: '{file_name}'")
                        
                        for page_num, page in enumerate(doc):
                            page_text = page.get_text()
                            
                            # Se extração normal falhar, usar OCR
                            if not page_text.strip():
                                print(f"    🔍 OCR página {page_num + 1}")
                                ocr_text = self._ocr_pdf_page(doc, page_num)
                                if ocr_text.strip():
                                    page_text = ocr_text
                            
                            if page_text.strip():
                                texto_completo += page_text + "\n\n"
                        
                        if texto_completo.strip():
                            documentos_carregados.append(
                                Document(
                                    page_content=texto_completo, 
                                    metadata={"source": caminho_arquivo_pdf, "cached": False}
                                )
                            )
                            
                            # Salvar no cache
                            with open(cache_file_path, "w", encoding="utf-8") as f:
                                f.write(texto_completo)
                            print(f"    💾 Salvo no cache")
                        
                        doc.close()
                    except Exception as e:
                        print(f"  ❌ Erro ao processar '{file_name}': {e}")
        
        print(f"✅ Total: {len(documentos_carregados)} documentos carregados")
        return documentos_carregados