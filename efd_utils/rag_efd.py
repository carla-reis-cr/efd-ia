import re
from pathlib import Path
from typing import List, Dict
import json
import shutil
import os
import time
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

def limpar_arquivos_temporarios():
    """Limpa arquivos temporários e modelos baixados."""
    try:
        # Limpa o cache do HuggingFace
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            print("Limpando cache do HuggingFace...")
            shutil.rmtree(cache_dir)
        
        # Limpa o vectorstore
        vectorstore_dir = Path(os.path.join(project_root, "efd_db"))
        print(vectorstore_dir)
        if vectorstore_dir.exists():
            print("Limpando vectorstore...")
            # Tenta remover várias vezes em caso de arquivo em uso
            for _ in range(3):
                try:
                    shutil.rmtree(vectorstore_dir)
                    break
                except PermissionError:
                    print("Aguardando arquivos serem liberados...")
                    time.sleep(1)

        # Limpa txt
        txt_file = Path(os.path.join(project_root, "data", "txt" , "validacoes_extraidas.md"))
        print(txt_file)
        if txt_file.exists():
            print("Limpando arquivo txt...")
            txt_file.unlink()

        # Limpa os chunks
        chunks_file = Path(os.path.join(project_root, "data", "structured" , "chunks.json"))
        print(chunks_file)
        if chunks_file.exists():
            print("Limpando arquivo de chunks...")
            chunks_file.unlink()
        
        # Limpa os arquivos JSON na pasta json_dir
        json_dir = Path(os.path.join(project_root, "data" , "structured" , "json"))
        print(json_dir)
        if json_dir.exists() and json_dir.is_dir():
            print("Limpando arquivos JSON na pasta json...")
            for json_file in json_dir.glob("*.json"):
                try:
                    json_file.unlink()
                except Exception as e:
                    print(f"⚠️ Não foi possível apagar o arquivo {json_file}: {str(e)}")

        print("✅ Limpeza concluída!")
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível limpar alguns arquivos: {str(e)}")

class EFDChunker:
    def __init__(self, markdown_path: str, model_name: str = "default-model"):
        self.markdown_path = Path(markdown_path)
        self.json_dir = Path(os.path.abspath(os.path.join(project_root, "data", "structured", "json")))
        self.chunks: List[Dict] = []
        self.model_name = model_name  # Armazena o nome do modelo como parâmetro

    def extract_registers(self) -> List[Dict]:
        """Extrai os registros do arquivo markdown e os organiza em chunks."""
        content = self.markdown_path.read_text(encoding='utf-8')
        
        # Padrão para encontrar registros
        pattern = r'## REGISTRO ([A-Z0-9]+):\s*([^#]+?)(?=## REGISTRO|$)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            register_code = match.group(1)
            register_content = match.group(2).strip()
            
            # Tenta carregar o JSON do registro
            json_path = self.json_dir / f"registro_{register_code}.json"
            campos = []
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if isinstance(json_data, list):
                            # Processa cada campo do registro
                            for campo in json_data:
                                campo_processado = {
                                    'numero': campo.get('Nº', ''),
                                    'codigo': campo.get('Campo', ''),
                                    'descricao': campo.get('Descrição', ''),
                                    'tipo': campo.get('Tipo', ''),
                                    'tamanho': campo.get('Tam', ''),
                                    'decimal': campo.get('Dec', ''),
                                    'entrada': campo.get('Entr', ''),
                                    'saida': campo.get('Saída', '')
                                }
                                campos.append(campo_processado)
                            print(f"✅ Campos carregados para o registro {register_code}: {len(campos)} campos")
                        else:
                            print(f"⚠️ JSON do registro {register_code} não está no formato esperado")
                except Exception as e:
                    print(f"❌ Erro ao carregar JSON do registro {register_code}: {str(e)}")
            
            # Cria um chunk para cada registro
            chunk = {
                'register_code': register_code,
                'content': register_content,
                'campos_exigidos': campos,
                'metadata': {
                    'source': str(self.markdown_path),
                    'register': register_code,
                    'has_json': len(campos) > 0,
                    'num_campos': len(campos),
                    'model_name': self.model_name  # Adiciona o modelo como metadado
                }
            }
            self.chunks.append(chunk)
            print(f"✅ Chunk criado para o registro {register_code} com {len(campos)} campos")
            
        return self.chunks
    
    def save_chunks(self, output_path: str):
        """Salva os chunks em um arquivo JSON."""
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ Chunks salvos em: {output_file}")

class EFDVectorStore:
    def __init__(self, chunks_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks_path = Path(chunks_path)
        
        # Configura o modelo de embeddings com base no parâmetro
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        self.vectorstore = None
        
    def create_vectorstore(self):
        """Cria o vectorstore a partir dos chunks."""
        # Carrega os chunks
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Converte chunks para documentos do LangChain
        documents = []
        for chunk in chunks:
            # Cria um texto que combina o conteúdo e os campos
            content = chunk['content']
            if chunk['campos_exigidos']:
                campos_text = "\n\nCampos do registro:\n"
                for campo in chunk['campos_exigidos']:
                    campos_text += f"- {campo['codigo']}: {campo['descricao']}\n"
                content += campos_text
            
            doc = Document(
                page_content=content,
                metadata=chunk['metadata']
            )
            documents.append(doc)
        
        print("Vectorstore: ", documents)
        # Cria o vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=os.path.join(project_root, "efd_db", "vectorstore")
        )
        print("✅ Vectorstore criado com sucesso!")
        
    def query(self, query: str, k: int = 3) -> List[Document]:
        """Realiza uma consulta no vectorstore."""
        if not self.vectorstore:
            raise ValueError("Vectorstore não foi criado. Execute create_vectorstore() primeiro.")
            
        return self.vectorstore.similarity_search(query, k=k)
    
    def close(self):
        """Fecha o vectorstore e libera recursos."""
        if self.vectorstore:
            self.vectorstore = None

class EFDChatTransformers:
    def __init__(self, model_name: str ):

        token= os.getenv('HUGGINGFACEHUB_API_TOKEN')

        login(token, new_session=False)

        # Carrega o modelo e tokenizer
        #model_name = "mistralai/Mistral-Nemo-Instruct-2407"
        #"neuralmind/bert-base-portuguese-cased"
        #"mistralai/Mistral-7B-Instruct-v0.2"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
            #trust_remote_code=True,
            #load_in_8bit=True,  # Usa quantização 8-bit para reduzir uso de memória
            #torch_dtype=torch.float16
        )
        
        # Cria o pipeline com parâmetros mais restritivos
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  # Aumentado para respostas mais completas
            temperature=0.3,     # Reduzido para respostas mais determinísticas
            top_p=0.9,          # Reduzido para maior foco
            repetition_penalty=1.2,  # Aumentado para evitar repetições
            do_sample=True,      # Habilita amostragem
            num_return_sequences=1,  # Retorna apenas uma sequência
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Cria o LLM do LangChain
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
    
    def __del__(self):
        """Limpa recursos quando o objeto é destruído."""
        # Libera memória do modelo
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'llm'):
            del self.llm
        
        # Força coleta de lixo
        import gc
        gc.collect()
    
    def format_prompt(self, query: str, context: List[Document]) -> str:
        """Formata o prompt com o contexto e a pergunta."""
        context_text = "\n\n".join([
            f"Registro: {doc.metadata['register']}\n"
            f"Conteúdo: {doc.page_content}"
            for doc in context
        ])
        
        prompt = f"""<|system|>
Você é um assistente especializado em EFD (Escrituração Fiscal Digital).
Sua tarefa é responder perguntas sobre registros e campos do EFD de forma clara e precisa.
Use APENAS as informações fornecidas no contexto para responder em Português do Brasil.
Se não souber a resposta, diga que não tem informações suficientes.

Contexto:
{context_text}
</s>
<|user|>
{query}
</s>
<|assistant|>"""
        return prompt
    
    def get_response(self, query: str, context: List[Document]) -> str:
        """Gera uma resposta usando o modelo de chat."""
        try:
            prompt = self.format_prompt(query, context)
            response = self.llm(prompt)
            
            # Extrai a resposta do texto gerado
            if isinstance(response, str):
                text = response
            elif isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict) and 'generated_text' in response[0]:
                    text = response[0]['generated_text']
                elif isinstance(response[0], str):
                    text = response[0]
                else:
                    return "Desculpe, não consegui gerar uma resposta adequada."
            else:
                return "Desculpe, não consegui gerar uma resposta adequada."
            
            # Limpa e formata a resposta
            response_text = text.split("<|assistant|>")[-1].strip()
            
            # Remove caracteres especiais e formatação indesejada
            response_text = re.sub(r'[^\w\s\.,;:!?()-]', '', response_text)
            response_text = re.sub(r'\s+', ' ', response_text).strip()
            
            # Verifica se a resposta faz sentido
            if len(response_text) < 10 or not any(c.isalpha() for c in response_text):
                return "Desculpe, não consegui gerar uma resposta adequada. Por favor, tente reformular sua pergunta."
            
            return response_text
            
        except Exception as e:
            print(f"Erro ao gerar resposta: {str(e)}")
            return "Desculpe, ocorreu um erro ao gerar a resposta. Por favor, tente novamente."


class EFDChatOllama:
    def __init__(self, model_name="llama3.2"):
            self.llm = ChatOllama(model=model_name)

    def format_prompt(self, query: str, context: List[Document]) -> str:
        context_text = "\n\n".join([
            f"Registro: {doc.metadata['register']}\n"
            f"Conteúdo: {doc.page_content}"
            for doc in context
        ])

        prompt = f"""Você é um assistente especializado em EFD (Escrituração Fiscal Digital).
            Responda com base apenas nas informações abaixo. Se não souber a resposta, diga que não tem informações suficientes.

            Contexto:
            {context_text}

            Pergunta: {query}
            Resposta:"""
        return prompt

    def get_response(self, query: str, context: List[Document]) -> str:
        prompt = self.format_prompt(query, context)
        response = self.llm([HumanMessage(content=prompt)])
        return response.content

class EFDChatBode:
    """
    Classe para interação com o modelo recogna-nlp/bode-7b-alpaca-pt-br-gguf via ctransformers.
    """
    def __init__(self, model_file=r"e:\Carla Reis\Projects\Notebooks\efd-ia\models\bode\bode-7b-alpaca-q8_0.gguf"):
        self.model_file = model_file

        self.template = """Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{instruction}

### Resposta:"""
        self.prompt = PromptTemplate(template=self.template, input_variables=["instruction"])

        self.llm = CTransformers(
            model="recogna-nlp/bode-7b-alpaca-pt-br-gguf",
            model_file=self.model_file,
            model_type='llama'
        )
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def format_context(self, context):
        # Junta o contexto dos documentos do vector store
        context_text = "\n\n".join([
            f"Registro: {doc.metadata.get('register', '')}\nConteúdo: {doc.page_content}"
            for doc in context
        ])
        return context_text

    def get_response(self, query, context_text ):
        # Monta a instrução com o contexto do vector store
        
        instruction = f"""Você é um assistente especializado em EFD (Escrituração Fiscal Digital).
Responda com base apenas nas informações abaixo. Se não souber a resposta, diga que não tem informações suficientes.

Contexto:
{context_text}

Pergunta: {query}"""
        response = self.llm_chain.run(instruction)
        return response.strip()
    
def main():
    try:
        # Limpa arquivos temporários ao finalizar
        limpar_arquivos_temporarios()
        
        # Cria os chunks
        chunker = EFDChunker(os.path.join(project_root,"data","txt","validacoes_extraidas.md"))
        chunks = chunker.extract_registers()
        chunker.save_chunks(os.path.join(project_root,"data","structured","chunks.json"))
        
        # Cria o vectorstore
        vectorstore = EFDVectorStore(os.path.join(project_root,"data","structured","chunks.json"))
        vectorstore.create_vectorstore()
        retriever = vectorstore.vectorstore.as_retriever()
        
        # Cria o chat
        #chat = EFDChatTransformers()
        
        chat = EFDChatOllama()
        

        # Exemplo de consulta
        #query = "Quais são os campos do registro C165?"
        #query = "Qual a regra do campo 01 do registro C165?"
        #query = "Qual a regra do campo 01 do registro C170?"
        #query = "Qual a regra do campo NUM_ITEM?"
        query = "Em quais registros usa o campo NUM_ITEM do registro C170?"

        #results = vectorstore.query(query)
        results = retriever.get_relevant_documents(query)
        
        # Gera a resposta
        response = chat.get_response(query, results)
        
        print("\nPergunta:", query)
        print("\nResposta:", response)
        
        
    finally:
        # Fecha o vectorstore antes de limpar
        if 'vectorstore' in locals():
            vectorstore.close()
        
        

if __name__ == "__main__":
    main()