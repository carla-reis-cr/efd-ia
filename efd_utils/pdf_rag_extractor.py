import json
import os
from typing import Dict, List, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import gc
import numpy as np  # Adicionando import explícito do numpy

class PDFRAGExtractor:
    def __init__(
        self,
        pdf_path: str,
        embedding_model: str,
        generation_model: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa o extrator RAG para PDFs.
        
        Args:
            pdf_path (str): Caminho para o arquivo PDF
            embedding_model (str): Nome do modelo de embeddings do Hugging Face
            generation_model (str): Nome do modelo de geração do Hugging Face
            device (str): Dispositivo para execução ('cuda' ou 'cpu')
        """
        self.pdf_path = pdf_path
        self.device = device
        
        # Inicializa o modelo de embeddings usando HuggingFaceEmbeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device}  # Usar o device passado (cpu ou cuda)
        )
        
        # Inicializa o modelo de geração com configurações otimizadas
        self.tokenizer = AutoTokenizer.from_pretrained(
            generation_model,
            use_fast=True
        )

        # Carregamento seguro do modelo de geração
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            generation_model,
            torch_dtype=torch.float32,  # Use float32 para CPU
            trust_remote_code=True
        )

        # Cria o pipeline de geração com configurações otimizadas
        self.generation_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=250,
            temperature=0.5,
            top_p=0.95
        )
        
        # Configura o text splitter com chunks menores
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        
        # Inicializa o vector store
        self.vector_store = None
        
    def load_and_split_pdf(self) -> List[str]:
        """Carrega o PDF e divide em chunks."""
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        texts = self.text_splitter.split_documents(pages)
        
        # Limpa a memória após o processamento
        del pages
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return texts
    
    def create_vector_store(self, texts: Optional[List[str]] = None) -> FAISS:
        """Cria ou atualiza o vector store com os chunks do PDF."""
        if texts is None:
            texts = self.load_and_split_pdf()
            
        # Cria o FAISS usando o modelo de embeddings do Langchain
        self.vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embedding_model
        )
                
        return self.vector_store
    
    def _create_prompt(self, context: str, question: str) -> str:
        """Cria o prompt para o modelo de geração."""
        return f"""Use o seguinte contexto para responder à pergunta:
Contexto: {context}

Pergunta: {question}

Sua resposta DEVE ser um objeto JSON válido contendo os campos extraídos. 
NÃO inclua nenhuma outra explicação ou texto antes ou depois do JSON.
Exemplo de formato esperado:
{{
  "campo1": "descrição1",
  "campo2": "descrição2"
}}

Resposta JSON:"""
    
    def extract_structured_data(self, query: str) -> Dict[str, Any]:
        """Extrai dados estruturados baseado na query."""
        if self.vector_store is None:
            self.create_vector_store()
            
        # Busca documentos relevantes
        docs = self.vector_store.similarity_search(query)  # Reduzido de 3 para 2
        context = "\n".join([doc.page_content for doc in docs])
        
        # Cria o prompt
        prompt = self._create_prompt(context, query)
        
        # Gera a resposta
        result = self.generation_pipeline(prompt)[0]['generated_text']
        
        # Limpa a memória
        del docs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return self._parse_result_to_json(result)
    
    def _parse_result_to_json(self, result: str) -> Dict[str, Any]:
        """Converte o resultado da extração para JSON."""
        try:
            # Tenta encontrar um bloco JSON no texto
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                return json.loads(json_str)
            return json.loads(result)
        except json.JSONDecodeError:
            # Se não for um JSON válido, tenta estruturar o texto
            lines = result.split('\n')
            structured_data = {}
            current_section = None
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    structured_data[key.strip()] = value.strip()
                elif line.strip():
                    if current_section is None:
                        current_section = line.strip()
                        structured_data[current_section] = []
                    else:
                        structured_data[current_section].append(line.strip())
            
            return structured_data
    
    def process_pdf(self, query: str) -> Dict[str, Any]:
        """Processa o PDF e retorna os dados estruturados."""
        return self.extract_structured_data(query)

def main():
    # Exemplo de uso
    pdf_path = os.path.join('data', 'pdfs', 'completo', 'efd_icms_ipi_3.1.7.pdf')
    
    # Inicializa o extrator com modelos personalizados
    extractor = PDFRAGExtractor(
        pdf_path=pdf_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        generation_model="google/flan-t5-xl",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Exemplo de query para extrair dados específicos
    query = "Extraia todos os campos e suas descrições do registro 0220"
    result = extractor.process_pdf(query)
    
    # Salva o resultado em um arquivo JSON
    output_path = os.path.join('data', 'structured', 'extracted_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Dados extraídos salvos em: {output_path}")

if __name__ == "__main__":
    main() 