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

Para configurar o ambiente necessário para executar os notebooks e scripts deste projeto, siga os passos abaixo. É recomendado o uso do gerenciador de ambientes Conda (Anaconda ou Miniconda).

**Pré-requisitos:**

*   [Anaconda](https://www.anaconda.com/products/distribution) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado e configurado no PATH do sistema.
*   (Opcional, para execução com GPU) Placa de vídeo NVIDIA com drivers compatíveis com CUDA 11.8 (ou a versão especificada em `environment.yml`).

**Passos para Configuração (Método Manual Recomendado):**

1.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/seu-usuario/projeto_efd_ia.git
    cd projeto_efd_ia
    ```

2.  **Crie o Arquivo `.env`:**
    *   Na raiz do projeto, crie um arquivo chamado `.env`.
    *   Adicione as seguintes variáveis de ambiente a este arquivo, substituindo os valores conforme necessário:
        ```dotenv
        # Caminho para o diretório contendo os arquivos PDF de entrada
        DIRECTORY_PDFS=data/pdfs/tests        # Diretório onde estão os PDFs
        OUTPUT_DIR_STRUCTURED=data/structured  # Diretório para saída dos dados estruturados

        # Adicione outras variáveis conforme necessário
        ```
    *   **Importante:** O script `config/env.py` valida a existência das variáveis marcadas como obrigatórias (atualmente `DIRECTORY_PDFS` e `OUTPUT_DIR_STRUCTURED`).

3.  **Crie o Ambiente Conda:**
    *   Abra um terminal (Prompt de Comando, Anaconda Prompt, etc.) na raiz do projeto.
    *   Execute o comando para criar o ambiente `efd-ia` a partir do arquivo `environment.yml`:
        ```bash
        conda env create -f environment.yml
        ```
    *   Este comando instalará Python, PyTorch, LangChain, Transformers e todas as outras dependências listadas no arquivo. Pode levar vários minutos.

4.  **Ative o Ambiente:**
    *   Após a criação bem-sucedida, ative o ambiente Conda:
        ```bash
        conda activate efd-ia
        ```
    *   Você deverá ver `(efd-ia)` no início do prompt do seu terminal, indicando que o ambiente está ativo.

5.  **Execute o Script de Configuração de Ambiente:**
    *   **Dentro do ambiente ativado (`efd-ia`)**, execute o seguinte script Python para validar as variáveis de ambiente:
        ```bash
        python config/env.py
        ```
        *(Este script carrega as variáveis do `.env` e verifica se as obrigatórias estão presentes)*

6.  **Pronto!** Seu ambiente está configurado e pronto para uso. Agora você pode executar os notebooks Jupyter ou outros scripts Python do projeto dentro deste ambiente ativado.

**Alternativa: Configuração Automatizada (Script Windows)**

Se preferir automatizar os passos 3, 4 e 5 em um único comando (no Windows), você pode usar o script `create_env.bat`:

1.  Siga os passos 1 e 2 acima (Clonar repositório e Criar `.env`).
2.  Execute o script batch na raiz do projeto:
    ```bash
    .\create_env.bat
    ```
3.  Após a conclusão, ative o ambiente manualmente se necessário:
    ```bash
    conda activate efd-ia
    ```

---

## Estrutura do Projeto (Exemplo)

```
projeto_efd_ia/
├── main.py                             # Executa a pipeline completa
│
├── requirements.txt                    # Dependências do projeto
│
├── README.md                           # Documentação do projeto
│
├── notebooks/                          # Análises e experimentos com Jupyter
│   ├── 01_texto_bruto/                 # Testes diretos com texto dos documentos
│   ├── 02_json_estruturado/            # Testes com registros estruturados
│   ├── 03_grafos/                      # Abordagens baseadas em grafos, (se necessário)
│   └── 04_avaliacoes/                  # Comparações e métricas agregadas
│
├── efd_rules/                          # Módulo de extração de regras
├── efd_struct/                         # Módulo de estruturação de dados 
├── efd_db/                             # Integração com banco de dados (não escopo)
├── efd_ai/                             # Visualizações e interface (não escopo)
├── efd_eval/                           # Métricas e geração de relatórios
├── efd_utils/                          # Utilitários e funções auxiliares
│
├── data/                               # Dados brutos, estruturados e relatórios
│
├── config/                             # Arquivos de configuração do projeto
│
├── notes/                              # Anotações relevantes
│
└── tests/                              # Testes automatizados (não escopo)
```

## Próximos Passos

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
