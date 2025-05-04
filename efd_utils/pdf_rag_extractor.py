import json
import os
import re
from typing import Dict, List, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    pipeline
)
import torch
import gc
import numpy as np
from dotenv import load_dotenv

class PDFRAGExtractor:
    def __init__(
        self,
        pdf_path: str,
        embedding_model: str = "neuralmind/bert-base-portuguese-cased",
        generation_model: str = "google/flan-t5-base",
        device: str = "cpu"  # Forçando uso de CPU por padrão
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
        
        # Inicializa o modelo de embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                'device': device
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        
        # Carrega a configuração do modelo para determinar seu tipo
        config = AutoConfig.from_pretrained(generation_model, trust_remote_code=True)
        
        # Inicializa o tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            generation_model,
            use_fast=True,
            model_max_length=1024,  # Reduzido para evitar problemas de memória
            trust_remote_code=True
        )
        
        # Determina o tipo de modelo e inicializa apropriadamente
        if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
            print("Usando modelo seq2seq (encoder-decoder)")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                generation_model,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
        else:
            print("Usando modelo causal")
            self.model = AutoModelForCausalLM.from_pretrained(
                generation_model,
                device_map="auto",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)

        # Configurações de geração otimizadas
        self.generation_config = {
            "max_length": 1024,
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.1,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }
        
        # Adiciona configurações específicas para modelos seq2seq
        if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
            self.generation_config.update({
                "forced_bos_token_id": self.tokenizer.bos_token_id,
                "forced_eos_token_id": self.tokenizer.eos_token_id
            })
        
        # Configura o text splitter com chunks maiores
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Inicializa o vector store
        self.vector_store = None
        
    def load_and_split_pdf(self) -> List[str]:
        """Carrega o PDF e divide em chunks."""
        print(f"\nCarregando PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        print(f"Total de páginas carregadas: {len(pages)}")
        
        # Imprime o conteúdo de cada página para debug
        print("\nConteúdo das páginas:")
        for i, page in enumerate(pages, 1):
            print(f"\nPágina {i}:")
            print(page.page_content[:500] + "...")
            print("-" * 80)
        
        texts = self.text_splitter.split_documents(pages)
        print(f"Total de chunks gerados: {len(texts)}")
        
        # Imprime alguns exemplos de chunks
        print("\nExemplos de chunks gerados:")
        for i, text in enumerate(texts[:3], 1):
            print(f"\nChunk {i}:")
            print(text.page_content)
            print("-" * 80)
        
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
            
        print("\nCriando vector store...")
        # Cria o FAISS usando o modelo de embeddings do Langchain
        self.vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        print("Vector store criado com sucesso!")
                
        return self.vector_store
    
    def _create_prompt(self, query: str) -> str:
        """
        Cria o prompt para o modelo de geração.
        """
        if self.vector_store is None:
            self.create_vector_store()
            
        print(f"\nBuscando documentos relevantes para a query: {query}")
        
        # Busca documentos relevantes
        docs = self.vector_store.similarity_search_with_score(
            query,
            k=12  # Aumentado para capturar mais contexto
        )
        
        if not docs:
            print("AVISO: Nenhum documento relevante encontrado!")
            return "Nenhum documento relevante encontrado."
        
        # Imprime os documentos relevantes encontrados com seus scores
        print(f"\nDocumentos relevantes encontrados ({len(docs)}):")
        for i, (doc, score) in enumerate(docs, 1):
            print(f"\nDocumento {i} (Score: {score:.4f}):")
            print(f"Conteúdo: {doc.page_content}")
            print("-" * 80)
        
        # Filtra documentos com score acima do threshold
        filtered_docs = [doc for doc, score in docs if score > 0.05]
        
        if not filtered_docs:
            print("AVISO: Nenhum documento com score suficiente encontrado!")
            return "Nenhum documento com score suficiente encontrado."
        
        context = "\n---\n".join([doc.page_content for doc in filtered_docs])
        
        # Imprime o contexto completo que será enviado ao modelo
        print("\nContexto completo enviado ao modelo:")
        print(context)
        print("\n" + "="*80 + "\n")
        
        return f"""Analise o texto abaixo e extraia as informações do registro 0200 do EFD ICMS/IPI.
Retorne APENAS um objeto JSON com a seguinte estrutura:

{{
    "cod_item": "string",
    "descr_item": "string",
    "cod_barra": "string",
    "cod_ant_item": "string",
    "unid_inv": "string",
    "tipo_item": "string",
    "cod_ncm": "string",
    "ex_ipi": "string",
    "cod_gen": "string",
    "cod_lst": "string",
    "aliq_icms": "string"
}}

Regras:
1. Retorne APENAS o JSON, sem texto adicional
2. Use os valores exatos encontrados no texto
3. Mantenha a estrutura exata do JSON
4. Inclua todos os campos encontrados
5. Se um campo não for encontrado, use string vazia

Texto para análise:
{context}"""
    
    def extract_structured_data(self, query: str) -> Dict:
        """
        Extrai dados estruturados do PDF usando RAG.
        """
        # Cria o prompt com o contexto relevante
        prompt = self._create_prompt(query)
        
        # Imprime o prompt completo para debug
        print("\nPrompt completo:")
        print(prompt)
        print("\n" + "="*80 + "\n")
        
        # Gera o texto usando o modelo com configurações mais restritas
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Move os inputs para o dispositivo correto
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Gera o texto
        outputs = self.model.generate(
            **inputs,
            **self.generation_config
        )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Imprime o resultado bruto do modelo
        print("\nResultado bruto do modelo:")
        print(result)
        print("\n" + "="*80 + "\n")
        
        # Tenta extrair o JSON do resultado
        try:
            # Procura por um bloco JSON no texto
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                json_str = json_match.group(0)
                # Limpa o JSON
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                # Tenta fazer o parse
                return json.loads(json_str)
            else:
                print("Nenhum JSON encontrado no resultado")
                return {}
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
            print(f"Texto que causou o erro: {result}")
            return {}
    
    def process_pdf(self, query: str) -> Dict[str, Any]:
        """Processa o PDF e retorna os dados estruturados."""
        return self.extract_structured_data(query)

def main():
    # Carrega as variáveis de ambiente
    load_dotenv()

    # Define o caminho do PDF
    pdf_path = os.getenv('PDF_COMPLETO')
    if not pdf_path:
        pdf_path = os.path.join('..', 'data', 'pdfs', 'completo', 'efd_icms_ipi_3.1.7.pdf')

    print(f"Usando PDF: {pdf_path}")

    # Inicialize o extrator com o caminho do PDF
    extractor = PDFRAGExtractor(
        pdf_path=pdf_path,
        embedding_model="neuralmind/bert-base-portuguese-cased",
        generation_model="google/flan-t5-base",  # Modelo mais leve
        device="cpu"  # Forçando uso de CPU
    )

    # Fazer queries com termos mais específicos
    query = "REGISTRO 0200 TABELA DE IDENTIFICAÇÃO DO ITEM PRODUTO E SERVIÇOS"
    try:
        resultado = extractor.process_pdf(query)
        print("\nResultado da extração:")
        print(json.dumps(resultado, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"\nErro durante a extração: {str(e)}")

if __name__ == "__main__":
    main() 