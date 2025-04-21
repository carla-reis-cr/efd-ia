import pandas as pd
import pdfplumber
import os
import json
from datetime import datetime
import re
import sys

# Adiciona o diretório pai ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efd_utils.document_reader import list_pdf_files  # Importa a função do novo arquivo
from config import env

def extract_pdf_content(pdf_path):
    """
    Extrai tabelas e texto de um PDF usando pdfplumber
    """
    print(f"Processando o arquivo: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extrair todas as tabelas
            tables = []
            text_content = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"\nProcessando página {page_num}")
                
                # Extrair tabelas
                page_tables = page.extract_tables()
                if page_tables:
                    print(f"Encontradas {len(page_tables)} tabelas na página {page_num}")
                    tables.extend(page_tables)
                
                # Extrair texto usando diferentes estratégias
                page_text = ""
                
                # Tenta extrair texto diretamente
                direct_text = page.extract_text()
                if direct_text:
                    page_text = direct_text
                else:
                    # Se não conseguir, tenta extrair por palavras
                    words = page.extract_words()
                    if words:
                        page_text = " ".join(word['text'] for word in words)
                    else:
                        # Se ainda não conseguir, tenta extrair por caracteres
                        chars = page.extract_text(x_tolerance=3, y_tolerance=3)
                        if chars:
                            page_text = chars
                
                if page_text:
                    print(f"Texto extraído da página {page_num}:")
                    print("-" * 50)
                    print(page_text[:200] + "..." if len(page_text) > 200 else page_text)
                    print("-" * 50)
                    text_content.append(page_text)
                else:
                    print(f"Nenhum texto encontrado na página {page_num}")
            
            print(f"\nTotal de tabelas extraídas: {len(tables)}")
            print(f"Total de páginas com texto: {len(text_content)}")
            
            return tables, text_content
            
    except Exception as e:
        print(f"Erro ao processar o arquivo {pdf_path}: {str(e)}")
        return [], []

def extract_field_explanations(text):
    """
    Extrai as explicações dos campos que aparecem após a tabela
    """
    explanations = {}
    # Padrão para encontrar explicações de campos
    pattern = r'Campo\s+(\d+)\s*\(([^)]+)\)\s*-\s*([^.]+)'
    
    matches = re.finditer(pattern, text)
    for match in matches:
        field_num = match.group(1)
        field_name = match.group(2)
        explanation = match.group(3).strip()
        explanations[field_num] = {
            "campo": field_name,
            "explicacao": explanation
        }
    
    return explanations

def extract_field_validations(text):
    """
    Extrai as validações e valores válidos para cada campo do texto
    """
    validations = {}
    
    # Padrão para encontrar as informações de campo com suas validações
    campo_pattern = r'Campo\s+(\d+)\s*\(([^)]+)\)\s*[-–]\s*(?:Validação:|Valor[es]* Válido[s]*:|\s*Preenchimento:)([^.]*(?:\.[^C][^.]*)*)'
    
    # Encontra todas as ocorrências
    matches = re.finditer(campo_pattern, text, re.IGNORECASE | re.MULTILINE)
    
    for match in matches:
        campo_num = match.group(1)
        campo_nome = match.group(2)
        validacao_text = match.group(3).strip()
        
        # Separa valores válidos e validações
        valor_valido = None
        validacao = None
        
        # Procura por "Valor(es) Válido(s):"
        valor_match = re.search(r'Valor(?:es)?\s+Válido(?:s)?:\s*\[(.*?)\]', validacao_text)
        if valor_match:
            valor_valido = valor_match.group(1)
        
        # Procura por "Validação:"
        validacao_match = re.search(r'Validação:\s*(.*?)(?:\.|$)', validacao_text)
        if validacao_match:
            validacao = validacao_match.group(1).strip()
        
        # Procura por "Preenchimento:"
        preenchimento_match = re.search(r'Preenchimento:\s*(.*?)(?:\.|$)', validacao_text)
        if preenchimento_match:
            if validacao:
                validacao += "; " + preenchimento_match.group(1).strip()
            else:
                validacao = preenchimento_match.group(1).strip()
        
        validations[campo_num] = {
            "campo": campo_nome,
            "valor_valido": valor_valido,
            "validacao": validacao
        }
    
    return validations

def clean_text(text):
    """
    Limpa o texto removendo \n e tratando espaços adequadamente
    """
    if not isinstance(text, str):
        return text
        
    # Substitui \n entre letras por espaço
    text = re.sub(r'([a-zA-Z])\n([a-zA-Z])', r'\1 \2', text)
    # Remove \n restantes
    text = text.replace('\n', ' ')
    # Remove aspas duplas escapadas (corrigido)
    text = text.replace('\\"', '"')
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_additional_info(text):
    """
    Extrai informações adicionais como Nível hierárquico e Ocorrência
    """
    info = {}
    
    # Padrões para buscar as informações com diferentes separadores
    nivel_pattern = r'Nível\s+hierárquico\s*[-:]\s*(\d+)'
    ocorrencia_pattern = r'Ocorrência\s*[-:]\s*([^.\n]+)'
    
    # Busca nível hierárquico
    nivel_match = re.search(nivel_pattern, text, re.IGNORECASE)
    if nivel_match:
        info['nivel_hierarquico'] = int(nivel_match.group(1))
    
    # Busca ocorrência
    ocorrencia_match = re.search(ocorrencia_pattern, text, re.IGNORECASE)
    if ocorrencia_match:
        # Limpa e padroniza o texto da ocorrência
        ocorrencia = clean_text(ocorrencia_match.group(1))
        # Remove parênteses extras se existirem
        ocorrencia = re.sub(r'\(|\)', '', ocorrencia)
        info['ocorrencia'] = ocorrencia
    
    return info

def extract_relationships(text, registro_atual):
    """
    Extrai relacionamentos entre registros e campos mencionados no texto
    """
    relationships = []
    registros_relacionados = set()  # Set para controlar registros já relacionados
    
    # Padrões para encontrar referências a registros
    registro_refs = [
        # Referência direta a registro
        r'registro\s+(\d{4})',
        # Referência a campo de outro registro
        r'campo\s+(\w+)\s+do\s+registro\s+(\d{4})',
        # Referência "deve existir no registro"
        r'deve\s+existir\s+no\s+registro\s+(\d{4})',
        # Referência "informar no registro"
        r'informar\s+no\s+(\d{4})',
        # Referência a registros entre parênteses
        r'\((?:0*(\d{1,4})(?:\s+ou\s+0*(\d{1,4}))+)\)',
    ]
    
    for pattern in registro_refs:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            if '(' in pattern:  # Caso especial para registros entre parênteses
                # Pega todos os números do grupo
                all_numbers = re.findall(r'\d+', match.group(0))
                for reg_ref in all_numbers:
                    # Padroniza para 4 dígitos
                    reg_ref = reg_ref.zfill(4)
                    if reg_ref != registro_atual and reg_ref not in registros_relacionados:
                        relationships.append({
                            "tipo": "registro_relacionado",
                            "registro": reg_ref
                        })
                        registros_relacionados.add(reg_ref)
            elif len(match.groups()) == 1:
                reg_ref = match.group(1)
                if reg_ref != registro_atual and reg_ref not in registros_relacionados:
                    relationships.append({
                        "tipo": "registro_relacionado",
                        "registro": reg_ref
                    })
                    registros_relacionados.add(reg_ref)
            elif len(match.groups()) == 2:
                campo_ref = match.group(1)
                reg_ref = match.group(2)
                if reg_ref != registro_atual and reg_ref not in registros_relacionados:
                    relationships.append({
                        "tipo": "campo_relacionado",
                        "registro": reg_ref,
                        "campo": campo_ref
                    })
                    registros_relacionados.add(reg_ref)
    
    return relationships

def process_table_to_json(table, section_text="", registro_atual=None):
    """
    Converte uma tabela em formato JSON organizando e tratando quebras de linha nas colunas
    """
    if not table:
        return None
    
    # Converte para DataFrame
    df = pd.DataFrame(table)
    
    # Remove linhas vazias
    df = df.dropna(how='all')
    
    # Remove colunas vazias
    df = df.dropna(axis=1, how='all')
    
    # Lista de possíveis cabeçalhos
    possible_headers = [
        ['Nº', 'Campo', 'Descrição', 'Tipo', 'Tam', 'Dec', 'Obrig'],
        ['Nº', 'Campo', 'Descrição', 'Tipo', 'Tam', 'Dec', 'Entr', 'Saída'],
        ['Nº', 'Campo', 'Descrição', 'Tipo', 'Tam', 'Dec', 'Entrada', 'Saída'],
        ['Nº', 'Campo', 'Descrição', 'Tipo', 'Tam', 'Dec', 'Entr/Saída'],
        ['Nº', 'Campo', 'Descrição', 'Tipo', 'Tam', 'Dec', 'Entrada/Saída']
    ]
    
    # Procura pela linha que contém os cabeçalhos
    header_row = None
    selected_headers = None
    
    for idx, row in df.iterrows():
        row_str = ' '.join(str(cell) for cell in row if pd.notna(cell))
        for headers in possible_headers:
            if all(header.lower() in row_str.lower() for header in headers):
                header_row = idx
                selected_headers = headers
                break
        if header_row is not None:
            break
    
    if header_row is not None:
        try:
            # Encontra as colunas corretas baseado no cabeçalho
            header_cells = df.iloc[header_row]
            selected_cols = []
            
            for header in selected_headers:
                for col_idx, cell in enumerate(header_cells):
                    if pd.notna(cell) and header.lower() in str(cell).lower():
                        selected_cols.append(col_idx)
                        break
            
            if len(selected_cols) == len(selected_headers):
                # Seleciona apenas as colunas necessárias
                df = df.iloc[:, selected_cols]
                
                # Define os cabeçalhos
                df.columns = selected_headers
                
                # Remove a linha de cabeçalho
                df = df.iloc[header_row + 1:].copy()
                
                # Inicializa variáveis para controle de continuação
                current_record = None
                records = []
                
                # Processa as linhas, tratando continuações
                for idx, row in df.iterrows():
                    if pd.notna(row['Nº']) and row['Nº'].strip():  # Nova entrada
                        if current_record:
                            records.append(current_record)
                        current_record = row.to_dict()
                    elif current_record is not None:  # Continuação da linha anterior
                        # Concatena os valores não vazios com a linha anterior
                        for col in df.columns:
                            if pd.notna(row[col]) and str(row[col]).strip():
                                current_value = str(current_record[col])
                                new_value = str(row[col]).strip()
                                current_record[col] = f"{current_value} {new_value}".strip()
                
                # Adiciona o último registro
                if current_record:
                    records.append(current_record)
                
                # Remove registros vazios ou inválidos
                records = [record for record in records if any(str(v).strip() for v in record.values())]
                
                # Limpa e formata os valores dos registros
                for record in records:
                    for key in record:
                        if isinstance(record[key], str):
                            # Limpa o texto primeiro
                            cleaned_value = clean_text(record[key])
                            
                            # Tratamento especial para o campo Dec
                            if key == 'Dec' and (not cleaned_value or cleaned_value == '-'):
                                record[key] = "NA"
                            else:
                                record[key] = cleaned_value
                
                # Extrai as informações dos campos do texto da seção
                for record in records:
                    field_num = record['Nº']
                    field_name = record['Campo']
                    
                    # Padrão para encontrar informações do campo
                    campo_pattern = rf"Campo\s+{field_num}\s*\({field_name}\)\s*[-–]\s*((?:.|[\n])*?)(?=Campo\s+\d+\s*\(|$)"
                    campo_match = re.search(campo_pattern, section_text, re.IGNORECASE)
                    
                    if campo_match:
                        campo_text = clean_text(campo_match.group(1).strip())
                        
                        # Extrai valor válido - padrão ajustado para capturar mais variações
                        valor_valido_match = re.search(r'Valores?\s+[Vv]álidos?(?:\s+)?:\s*\[(.*?)\]', campo_text)
                        if valor_valido_match:
                            record['valor_valido'] = clean_text(valor_valido_match.group(1))
                        
                        # Extrai validação
                        validacao_match = re.search(r'Validação:\s*(.*?)(?=(?:Preenchimento:|Valores?\s+[Vv]álidos?:|$))', campo_text, re.DOTALL)
                        if validacao_match:
                            record['validacao'] = clean_text(validacao_match.group(1))
                        
                        # Extrai preenchimento
                        preenchimento_match = re.search(r'Preenchimento:\s*(.*?)(?=(?:Validação:|Valores?\s+[Vv]álidos?:|$))', campo_text, re.DOTALL)
                        if preenchimento_match:
                            record['preenchimento'] = clean_text(preenchimento_match.group(1))
                
                # Extrai informações adicionais do texto da seção
                additional_info = extract_additional_info(section_text)
                
                # Extrai relacionamentos do texto da seção
                relationships = []
                
                # Para cada campo, procura relacionamentos específicos
                for record in records:
                    field_num = record['Nº']
                    field_name = record['Campo']
                    
                    # Procura no texto específico do campo
                    campo_pattern = rf"Campo\s+{field_num}\s*\({field_name}\)\s*[-–]\s*((?:.|[\n])*?)(?=Campo\s+\d+\s*\(|$)"
                    campo_match = re.search(campo_pattern, section_text, re.IGNORECASE)
                    
                    if campo_match:
                        campo_text = campo_match.group(1)
                        field_relationships = extract_relationships(campo_text, registro_atual)
                        if field_relationships:
                            record['relacionamentos'] = field_relationships
                
                # Procura relacionamentos gerais na seção
                section_relationships = extract_relationships(section_text, registro_atual)
                
                # Cria o objeto final com todas as informações
                final_data = {
                    'nivel_hierarquico': additional_info.get('nivel_hierarquico'),
                    'ocorrencia': additional_info.get('ocorrencia'),
                    'relacionamentos': section_relationships,
                    'campos': records
                }
                
                return final_data
                
        except Exception as e:
            print(f"Erro ao processar tabela: {str(e)}")
    
    return None

def extract_registro_number(text_content):
    """
    Extrai o número do registro do texto
    """
    registros = []
    for text in text_content:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if "registro" in line.lower():
                # Tenta extrair o número do registro
                numbers = re.findall(r'\d+', line)
                if numbers:
                    registros.append(numbers[0])  # Armazena todos os números encontrados
    return registros  # Retorna todos os números de registro encontrados

def extract_registro_sections(text_content):
    """
    Extrai as seções de cada registro, começando com 'REGISTRO XXXX:' e 
    capturando todo o conteúdo até o próximo título de registro
    """
    registros = []
    registros_processados = set()
    
    # Juntar todo o texto em um único conteúdo
    full_text = '\n'.join(text_content)
    
    # Padrões para identificar o início de um registro
    registro_patterns = [
        r'REGISTRO\s+([A-Z0-9]+)\s*:\s*([^\n]+)',  # Padrão original
        r'Registro\s+([A-Z0-9]+)\s*[:-]\s*([^\n]+)',  # Variação com R maiúsculo/minúsculo
        r'BLOCO\s+([A-Z0-9]+)\s*:\s*([^\n]+)',  # Padrão para blocos
        r'Bloco\s+([A-Z0-9]+)\s*[:-]\s*([^\n]+)',  # Variação de blocos
    ]
    
    # Combina todos os padrões em uma única expressão regular
    combined_pattern = '|'.join(f'({pattern})' for pattern in registro_patterns)
    
    print("\nProcurando registros no texto...")
    print("-" * 50)
    print(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)
    print("-" * 50)
    
    # Encontra todas as ocorrências de registros
    matches = list(re.finditer(combined_pattern, full_text, re.MULTILINE | re.IGNORECASE))
    
    if not matches:
        print("Nenhum registro encontrado no texto!")
        return []
    
    print(f"Encontrados {len(matches)} possíveis registros")
    
    # Processa cada registro encontrado
    for i, match in enumerate(matches):
        # Encontra o grupo que deu match (pode ser qualquer um dos padrões)
        for group_idx, group in enumerate(match.groups()):
            if group is not None and re.search(r'[A-Z0-9]+', group):
                # Extrai o número do registro e o título
                registro_match = re.search(r'(?:REGISTRO|Registro|BLOCO|Bloco)\s+([A-Z0-9]+)\s*[:-]\s*([^\n]+)', group)
                if registro_match:
                    registro_numero = registro_match.group(1)
                    registro_titulo = registro_match.group(2)
                    
                    # Define o início do conteúdo como o fim do match atual
                    content_start = match.end()
                    
                    # Define o fim do conteúdo como o início do próximo match ou o fim do texto
                    content_end = matches[i + 1].start() if i < len(matches) - 1 else len(full_text)
                    
                    # Extrai o conteúdo entre os matches
                    content = full_text[content_start:content_end].strip()
                    
                    # Limpa o conteúdo
                    cleaned_content = []
                    for line in content.split('\n'):
                        line = line.strip()
                        # Pula linhas que começam com Nº Campo Descrição Tipo Tam Dec Obrig
                        if re.match(r'^\s*Nº\s+Campo\s+Descrição\s+Tipo\s+Tam\s+Dec\s+(?:Obrig|Entr|Saída|Entrada|Saída)', line, re.IGNORECASE):
                            continue
                        # Pula linhas que começam com números seguidos de espaços (linhas da tabela)
                        if re.match(r'^\s*\d+\s+', line):
                            continue
                        # Mantém apenas linhas com texto descritivo
                        if line:
                            cleaned_content.append(line)
                    
                    if registro_numero not in registros_processados:
                        registros.append({
                            "numero": registro_numero,
                            "titulo": registro_titulo.strip(),
                            "conteudo": '\n'.join(cleaned_content)
                        })
                        registros_processados.add(registro_numero)
                        print(f"Registro {registro_numero} processado")
                break
    
    print(f"Total de registros extraídos: {len(registros)}")
    return registros

def clean_json_string(obj):
    """
    Limpa strings no objeto antes de converter para JSON
    """
    if isinstance(obj, dict):
        return {k: clean_json_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_string(item) for item in obj]
    elif isinstance(obj, str):
        # Remove aspas escapadas e limpa o texto
        return obj.replace('\\"', '"')
    return obj

def main():
    print("Iniciando o script...")
    # Lista todos os PDFs na pasta
    pdf_files = list_pdf_files()
    print(f"Arquivos PDF encontrados: {pdf_files}")
    
    if not pdf_files:
        print("Nenhum arquivo PDF encontrado na pasta documents/pdfs")
        return
    
    # Dicionário para armazenar todos os registros
    all_registros = {
        "registros": []
    }
    
    # Set para controlar registros já processados
    registros_processados = set()
    
    print(f"Arquivos PDF encontrados: {len(pdf_files)}")
    for pdf_file in pdf_files:
        print(f"\nProcessando: {pdf_file}")
        try:
            # Extrair conteúdo
            print("Extraindo conteúdo do PDF...")
            tables, text_content = extract_pdf_content(pdf_file)
            print(f"Total de tabelas encontradas: {len(tables)}")
            print(f"Total de páginas processadas: {len(text_content)}")
            
            # Extrai as seções dos registros
            print("Extraindo seções dos registros...")
            registros = extract_registro_sections(text_content)
            print(f"Total de registros encontrados: {len(registros)}")
            
            # Mapeia as tabelas para os registros correspondentes
            tabelas_processadas = set()
            
            for registro in registros:
                registro_numero = registro["numero"]
                print(f"\nProcessando registro {registro_numero}...")
                
                # Verifica se este registro já foi processado
                if registro_numero in registros_processados:
                    print(f"Registro {registro_numero} já processado anteriormente. Verificando por informações complementares...")
                    continue
                
                # Cria o dicionário base com a ordem desejada
                registro_data = {
                    "numero": registro_numero,
                    "titulo": clean_text(registro["titulo"]),
                    "conteudo": clean_text(registro["conteudo"]),
                    "nivel_hierarquico": None,
                    "ocorrencia": None,
                    "relacionamentos": [],
                    "campos": []
                }
                
                # Procura a tabela correspondente ao registro
                print(f"Procurando tabela para o registro {registro_numero}...")
                for i, table in enumerate(tables):
                    if i not in tabelas_processadas and table:
                        print(f"Processando tabela {i}...")
                        json_data = process_table_to_json(table, registro["conteudo"], registro_numero)
                        if json_data:
                            table_text = '\n'.join([str(row) for row in table])
                            if registro_numero in table_text:
                                registro_data["nivel_hierarquico"] = json_data.get('nivel_hierarquico')
                                registro_data["ocorrencia"] = json_data.get('ocorrencia')
                                registro_data["relacionamentos"] = json_data.get('relacionamentos', [])
                                registro_data["campos"] = json_data.get('campos', [])
                                tabelas_processadas.add(i)
                                print(f"Tabela encontrada para o registro {registro_numero}")
                            break
                
                all_registros["registros"].append(registro_data)
                registros_processados.add(registro_numero)
                print(f"Registro {registro_numero} adicionado ao JSON")
        
        except Exception as e:
            print(f"Erro ao processar o arquivo {pdf_file}: {str(e)}")
            continue
    
    # Gera timestamp para nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Limpa as strings antes de salvar
    print("Limpando strings antes de salvar...")
    all_registros = clean_json_string(all_registros)
    
    # Obtém o diretório de saída da variável de ambiente ou usa um valor padrão
    output_dir = os.getenv('OUTPUT_DIR_STRUCTURED', 'documents/structured')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"registros_combinados_{timestamp}.json")
    
    print(f"Salvando resultados em {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_registros, f, ensure_ascii=False, indent=4)
    
    print(f"\nTodos os registros foram combinados e salvos em: {output_file}")
    print(f"Total de registros processados: {len(registros_processados)}")

if __name__ == "__main__":
    print("Script iniciado")
    main()
    print("Script concluído")