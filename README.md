# 🧠 Projeto EFD-IA: Extração, Estruturação e Avaliação com Inteligência Artificial

Este projeto visa criar uma bateria de testes para avaliar a aplicação de modelos de IA sobre documentos fiscais, com foco no **EFD ICMS/IPI**. A solução é composta por etapas que envolvem extração de regras, estruturação de dados, integração com banco de dados e automação com IA.

---

## 📌 Funcionalidades Principais

**📄 Extração Automática de Regras** de manuais técnicos da Receita Federal (SPED)

**🔎 Validação Inteligente** de registros fiscais com regex e lógica condicional

**🤖 Explicações Contextuais** usando RAG (Retrieval-Augmented Generation)

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

## 🚀 Como Começar

```bash
# Clone o repositório
git clone https://github.com/carla-reis-cr/efd-ia.git
cd efd_ia

# Crie um ambiente Conda chamado efd-ia
conda create --name efd-ia python=3.10

# Ative o ambiente Conda
conda activate efd-ia

# Instale as dependências
pip install -r requirements.txt
```

### Passos Explicados:

1. **Criação do Ambiente Conda**: O comando `conda create --name efd-ia python=3.10` cria um novo ambiente Conda chamado `efd-ia` com Python 3.10.
2. **Ativação do Ambiente**: Use `conda activate efd-ia` para ativar o ambiente Conda recém-criado.
3. **Instalação das Dependências**: Com o ambiente ativado, instale as dependências listadas no `requirements.txt` usando o `pip`.


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
