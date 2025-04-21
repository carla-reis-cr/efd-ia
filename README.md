# 🧠 Projeto EFD-IA: Extração, Estruturação e Avaliação com Inteligência Artificial

Este projeto visa criar uma bateria de testes para avaliar a aplicação de modelos de IA sobre documentos fiscais, com foco no **EFD ICMS/IPI**. A solução é composta por etapas que envolvem extração de regras, estruturação de dados, integração com banco de dados e automação com IA.

---

## 📌 Objetivos

- Extrair regras e padrões de documentos fiscais brutos.
- Estruturar os dados em formatos padronizados (JSON, CSV).
- Avaliar a consistência das informações com modelos de IA.
- Automatizar o processo e integrar com sistemas ERP e banco de dados.

---

## 🚀 Como Começar

### Configuração do Ambiente

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/projeto_efd_ia.git
cd projeto_efd_ia
```

2. Crie um ambiente virtual e ative:
```bash
conda create --name efd-ia python=3.10
conda activate efd-ia
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Configuração das Variáveis de Ambiente

1. Copie o arquivo de exemplo `.env.example` para criar seu próprio arquivo `.env`:
```bash
cp .env.example .env
```

2. Edite o arquivo `.env` com suas configurações:
```ini
# Diretórios do projeto
DIRECTORY_PDFS=documents/pdfs/tests        # Diretório onde estão os PDFs
OUTPUT_DIR_STRUCTURED=documents/structured  # Diretório para saída dos dados estruturados

# Adicione outras variáveis conforme necessário
```

3. Certifique-se de que os diretórios especificados existam:
```bash
mkdir -p documents/pdfs/tests documents/structured
```

> **Nota**: O arquivo `.env` não é versionado no Git por conter informações sensíveis. Sempre mantenha suas credenciais e configurações locais neste arquivo.

---

## 🧱 Estrutura do Projeto

projeto_efd_ia/ ├── main.py 

projeto_efd_ia/
├── main.py # Executa a pipeline completa

├── requirements.txt # Dependências do projeto

├── README.md # Documentação do projeto

│

├── notebooks/ # Análises e experimentos com Jupyter

│ ├── 01_texto_bruto.ipynb # Extração e exploração inicial

│ ├── 02_texto_estruturado.ipynb # Validação e estruturação

│ └── 03_teste_com_ia.ipynb # Avaliações com modelos IA

│

├── efd_rules/ # Módulo de extração de regras

├── efd_struct/ # Módulo de estruturação de dados

├── efd_db/ # Integração com banco de dados

├── efd_ai/ # Avaliação e automação com IA

├── efd_eval/ # Métricas e geração de relatórios

├── efd_utils/ # Utilitários e funções auxiliares

├── data/ # Dados brutos, estruturados e relatórios

├── config/ # Arquivos de configuração

└── tests/ # Testes automatizados

---

## 🧪 Etapas de Testes

1. **Texto Bruto**
   - Leitura de documentos
   - Identificação de padrões
   - Processamento de texto

2. **Texto Estruturado**
   - Geração de dicionários
   - Exportação de dados estruturados
   - Validação dos formatos

3. **Aplicação com IA**
   - Verificação de inconsistências
   - Classificação automática com modelos treinados
   - Integração com sistemas externos (ERP, banco de dados)

---

## 🧠 Tecnologias Utilizadas

- Python 3.10+

- Jupyter Notebook

- pandas, spaCy, scikit-learn, transformers

- PostgreSQL ou SQLite (opcional)

- pytest (testes unitários)


## 📊 Resultados Esperados

- Dados estruturados e validados com base em documentos fiscais reais para futura aplicação em ambiente real promovendo a automação fiscal e integração ERP.

## 📂 Licença
Este projeto está sob a licença MIT.
