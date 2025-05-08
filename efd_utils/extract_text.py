import re
from pathlib import Path
import pandas as pd
from docling.document_converter import DocumentConverter
import json
import os
from dotenv import load_dotenv
load_dotenv()

output_dir = Path("data/structured/json")
tables = []

def sanitize_filename(filename):
    # Remove caracteres inválidos para nomes de arquivo
    invalid_chars = r'[<>:"/\\|?*\n\r\t]'
    
    # Remove acentos e caracteres especiais
    filename = re.sub(r'[áàâãéèêíïóôõöúçñºª]', '', filename)
    
    # Substitui caracteres especiais por underscore
    filename = re.sub(r'[^\w\s-]', '_', filename)
    
    # Substitui múltiplos espaços por um único espaço
    filename = re.sub(r'\s+', ' ', filename)
    
    # Remove caracteres inválidos
    filename = re.sub(invalid_chars, '_', filename)
    
    # Remove múltiplos underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Remove underscores no início e fim
    filename = filename.strip('_')
    
    # Limita o tamanho do nome do arquivo
    if len(filename) > 100:
        # Pega as primeiras palavras até chegar próximo a 100 caracteres
        words = filename.split()
        filename = words[0]
        for word in words[1:]:
            if len(filename) + len(word) + 1 <= 100:
                filename += '_' + word
            else:
                break
    
    return filename

def is_continuation_table(df: pd.DataFrame) -> bool:
    first_col = df.columns[0]
    try:
        first_values = df[first_col].dropna().head(3).astype(str)
        numbers = first_values.str.extract(r'^(\d+)$')[0].dropna()
        return len(numbers) < len(first_values)
    except Exception:
        return True

def fix_table_fragmented_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_num = next((col for col in df.columns if "Nº" in str(col) or "N º" in str(col)), None)
    col_campo = next((col for col in df.columns if "Campo" in str(col)), None)
    col_desc = next((col for col in df.columns if "Descrição" in str(col)), None)

    if not all([col_num, col_campo, col_desc]):
        return df

    rows_to_drop = []
    last_valid_idx = None

    for i in range(len(df)):
        num_val = str(df.at[i, col_num]) if pd.notna(df.at[i, col_num]) else ""
        is_valid_num = bool(re.match(r"^\d+$", num_val.strip()))
        campo_val = str(df.at[i, col_campo]) if pd.notna(df.at[i, col_campo]) else ""
        desc_val = str(df.at[i, col_desc]) if pd.notna(df.at[i, col_desc]) else ""

        if is_valid_num and campo_val.strip():
            last_valid_idx = i
        elif not is_valid_num or not campo_val.strip():
            if last_valid_idx is not None:
                if desc_val.strip():
                    old_desc = str(df.at[last_valid_idx, col_desc]) if pd.notna(df.at[last_valid_idx, col_desc]) else ""
                    df.at[last_valid_idx, col_desc] = (old_desc + " " + desc_val).strip()
                if campo_val.strip():
                    if not re.match(r"^[A-Z_]+$", campo_val.strip()) or " " in campo_val.strip():
                        df.at[last_valid_idx, col_desc] = (df.at[last_valid_idx, col_desc] + " " + campo_val.strip()).strip()
                rows_to_drop.append(i)

    df = df.drop(rows_to_drop).reset_index(drop=True)
    return df

def extrair_secoes(markdown):
    # Padrão para encontrar registros que começam com ##
    padrao = r'##\s*([^#]+?)(?=##|$)'
    matches = re.finditer(padrao, markdown, re.DOTALL)
    
    secoes = []
    for match in matches:
        registro = match.group(1).strip()
        if registro:
            secoes.append(f"## {registro}")
    
    return secoes, None

def extrair_texto(pdf_path):
    """Extrai apenas o texto do PDF e salva em markdown, removendo tabelas e todas as linhas em branco."""
    # Converter PDF para markdown
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(pdf_path)
    
    # Extrair markdown completo
    markdown_completo = conv_res.document.export_to_markdown()
    
    # Criar markdown limpo
    markdown_limpo = []
    limpa_linha = False
    
    for linha in markdown_completo.splitlines():
        limpa_linha = False
        
        # Se for uma seção de registro, mantém como está
        if linha.startswith('## REGISTRO'):
            markdown_limpo.append(linha)
            continue
            
        # Se for outra seção, remove o ##
        if linha.startswith('##'):
            linha = linha.replace('##', '').strip()
            
        if re.match(r'^\s*\|.*\|\s*$', linha):  # detecta linha de tabela
            limpa_linha = True
            continue
            
        if not linha.strip():
            limpa_linha = True
            
        if not limpa_linha:
            # Remove a barra invertida antes do sublinhado
            linha = re.sub(r'\\_', '_', linha)
            markdown_limpo.append(linha)
    
    # Incrementar no arquivo markdown
    incrementar_markdown(markdown_limpo)

def incrementar_markdown(markdown_limpo):
    markdown_path = Path("data/txt/validacoes_extraidas.md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)  # Garante que a pasta existe

    # Junta todas as linhas em uma única string com quebra de linha
    texto_a_adicionar = "\n".join(markdown_limpo) + "\n"

    # Abre o arquivo em modo append ('a') ou cria se não existir
    with open(markdown_path, "a", encoding="utf-8") as f:
        f.write(texto_a_adicionar)

def extrair_tabelas(pdf_path):
    """Extrai apenas as tabelas do PDF e salva em JSON, categorizando por registro."""
    # Criar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Converter PDF para markdown
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(pdf_path)
    
    # Extrair markdown completo para obter as seções
    markdown_completo = conv_res.document.export_to_markdown()
    secoes, _ = extrair_secoes(markdown_completo)
    
    # Processar tabelas
    tables = []
    merged_table = None
    
    for table in conv_res.document.tables:
        table_df = table.export_to_dataframe()
        
        if merged_table is None:
            merged_table = table_df
        else:
            if is_continuation_table(table_df):
                table_df.columns = merged_table.columns[:len(table_df.columns)]
                merged_table = pd.concat([merged_table, table_df], ignore_index=True)
            else:
                tables.append(merged_table)
                merged_table = table_df
    
    if merged_table is not None:
        tables.append(merged_table)
    
    # Corrigir tabelas fragmentadas
    tables_fixed = [fix_table_fragmented_rows(df) for df in tables]
    
    # Salvar tabelas em JSON
    for i, (df, secao) in enumerate(zip(tables_fixed, secoes)):
        # Extrair código do registro (ex: C170)
        codigo = re.search(r'REGISTRO\s+([A-Z0-9]+):', secao)
        if codigo:
            codigo = codigo.group(1)
            nome_arquivo = f"registro_{codigo}.json"
            json_path = output_dir / nome_arquivo
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json.loads(df.to_json(orient="records")), f, indent=2, ensure_ascii=False)
            print(f"✅ Tabela do registro {codigo} salva em: {json_path}")

def main():
    input_doc_path = Path(os.getenv('PDF_0200'))
    
    # Extrair texto
    extrair_texto(input_doc_path)
    
    # Extrair tabelas
    extrair_tabelas(input_doc_path)

if __name__ == "__main__":
    main() 