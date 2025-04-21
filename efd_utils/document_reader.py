import os

def list_pdf_files(directory="documents/pdfs"):
    """
    Lista todos os arquivos PDF em um diretório
    """
    pdf_files = []
    
    # Cria o diretório se não existir
    os.makedirs(directory, exist_ok=True)
    
    # Lista todos os arquivos .pdf no diretório
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    
    return pdf_files