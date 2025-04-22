import nltk
import os
import sys

def setup_nltk():
    # Definir o diretório de dados do NLTK no ambiente Conda
    conda_env_path = os.path.join(os.path.dirname(sys.executable), 'nltk_data')
    nltk.data.path.append(conda_env_path)

    # Criar o diretório se não existir
    os.makedirs(conda_env_path, exist_ok=True)

    # Baixar recursos necessários
    try:
        # Baixar o modelo de tokenização para português
        nltk.download('punkt', download_dir=conda_env_path)
        print('Modelo punkt baixado com sucesso!')
        
        # Baixar stopwords em português
        nltk.download('stopwords', download_dir=conda_env_path)
        print('Stopwords baixadas com sucesso!')
        
        # Baixar stemmer para português
        nltk.download('rslp', download_dir=conda_env_path)
        print('Stemmer RSLP baixado com sucesso!')
        
        # Baixar corpus em português
        nltk.download('mac_morpho', download_dir=conda_env_path)
        print('Corpus Mac-Morpho baixado com sucesso!')
        
        # Baixar o modelo de português do punkt
        nltk.download('portuguese', download_dir=conda_env_path)
        print('Modelo de português baixado com sucesso!')
        
        # Verificar se o modelo de português está disponível
        try:
            nltk.data.find('tokenizers/punkt/portuguese.pickle')
            print('Modelo de português encontrado!')
        except LookupError:
            print('Modelo de português não encontrado. Tentando baixar...')
            nltk.download('punkt', download_dir=conda_env_path)
            
    except Exception as e:
        print(f'Erro ao configurar NLTK: {str(e)}') 