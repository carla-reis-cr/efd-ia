import os
from dotenv import load_dotenv

def init_env():
    """
    Inicializa as variáveis de ambiente do projeto.
    Procura por arquivos .env na raiz do projeto.
    """
    # Encontra o diretório raiz do projeto (onde está o .env)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Carrega as variáveis de ambiente do arquivo .env
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)

    # Você pode adicionar validações de variáveis obrigatórias aqui
    required_vars = [
        # Liste aqui as variáveis de ambiente obrigatórias
        # 'API_KEY',
        # 'DATABASE_URL',
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"As seguintes variáveis de ambiente são obrigatórias mas não foram encontradas: {', '.join(missing_vars)}"
        )

# Carrega as variáveis de ambiente automaticamente ao importar o módulo
init_env() 