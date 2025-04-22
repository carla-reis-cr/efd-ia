# Metodologia do Sistema RAG para Consulta de Documentos Fiscais

## Objetivo

O objetivo deste projeto é desenvolver um sistema de Geração Aumentada por Recuperação (RAG - Retrieval-Augmented Generation) capaz de responder perguntas sobre documentos fiscais (como o Guia Prático EFD-ICMS/IPI) em formato PDF, indicando não apenas a resposta, mas também a localização específica da informação (registro e campo) dentro dos documentos originais.

## Fase 1: Preparação e RAG Inicial com Texto Bruto (`notebooks/01-texto-bruto/rag-text.ipynb`)

Nesta fase inicial, focamos em construir um pipeline RAG funcional usando o texto extraído diretamente dos PDFs, com uma estruturação básica por registros.

1.  **Extração de Texto e Estruturação Inicial:**
    *   Utilizamos a biblioteca `pdfplumber` para extrair o texto completo de cada arquivo PDF localizado no diretório configurado.
    *   Implementamos uma lógica para tentar segmentar o texto extraído em blocos correspondentes a "REGISTROS" (baseado na ocorrência da palavra "REGISTRO").
    *   Para cada bloco identificado, extraímos um `registro_id` (o código numérico/alfanumérico que segue a palavra "REGISTRO").
    *   Os textos segmentados (`registro_text`) e seus respectivos `registro_id`s foram armazenados em um DataFrame pandas para análise inicial do tamanho médio dos segmentos.

2.  **Criação de Embeddings e Vector Store:**
    *   Utilizamos o modelo `sentence-transformers/all-MiniLM-L6-v2` (via `HuggingFaceEmbeddings` da LangChain) para gerar embeddings vetoriais para cada bloco de `registro_text`.
    *   Cada bloco de texto foi encapsulado em um objeto `Document` da LangChain, adicionando o `registro_id` extraído anteriormente como metadado (`metadata={"registro_id": ...}`).
    *   Construímos um índice vetorial usando `FAISS` a partir dos `Document`s e seus embeddings, permitindo a busca eficiente por similaridade semântica.

3.  **Configuração do Retriever:**
    *   Criamos um `retriever` LangChain a partir do vector store FAISS, responsável por buscar os `Document`s (trechos de registro) mais relevantes para uma dada pergunta.

4.  **Configuração do LLM (Modelo de Linguagem):**
    *   Inicialmente, testamos com `HuggingFaceEndpoint` para acessar modelos via API.
    *   Devido a instabilidades ou limitações da API, migramos para a execução local do modelo `google/flan-t5-small` utilizando a biblioteca `transformers`.
    *   Carregamos o tokenizador (`T5Tokenizer`) e o modelo (`T5ForConditionalGeneration`) localmente.
    *   Configuramos um `pipeline` da `transformers` para a tarefa `text2text-generation`, especificando o modelo, tokenizador, parâmetros de geração (`max_new_tokens`, `temperature`, `top_p`) e o dispositivo de execução (GPU, se disponível, ou CPU).
    *   Utilizamos o wrapper `HuggingFacePipeline` da `langchain-huggingface` para integrar o pipeline local do `transformers` ao ecossistema LangChain, criando o objeto `llm`.

5.  **Montagem da Cadeia RAG (RetrievalQA):**
    *   Instanciamos uma cadeia `RetrievalQA` da LangChain, utilizando:
        *   O `llm` configurado (FLAN-T5 local).
        *   O `retriever` baseado em FAISS.
        *   O tipo de cadeia `stuff` (que insere todos os documentos recuperados diretamente no prompt do LLM).
        *   A opção `return_source_documents=True` para podermos inspecionar os documentos que o retriever considerou relevantes.

6.  **Teste Inicial:**
    *   Realizamos testes enviando perguntas para a `qa_chain` (ex: "Em quais registros o campo COD_ITEM é usado?").
    *   Analisamos a resposta gerada pelo LLM e os documentos fonte recuperados pelo retriever. Observamos que o sistema responde, mas a referência à fonte é limitada ao `registro_id` presente nos metadados.

## Fase 2: Refinamento para Citação de Fontes Estruturadas (Próximos Passos)

O RAG inicial funciona, mas não atende completamente ao requisito de citar campos específicos. Os próximos passos se concentram em melhorar a granularidade da informação indexada e a capacidade do LLM de usar essa informação.

1.  **Melhorar Extração e Estruturação de Dados:**
    *   **Desafio:** A segmentação atual por "REGISTRO" cria blocos de texto grandes e não estruturados internamente.
    *   **Ação:** Refinar o processo de extração do `pdfplumber` (ou explorar outras ferramentas como `Camelot` se houver muitas tabelas) para identificar não apenas registros, mas também *campos* dentro desses registros (ex: usando regex para encontrar descrições de campos ou analisando estruturas tabulares).
    *   **Objetivo:** Criar `Document`s mais granulares. Em vez de um documento por registro, podemos ter documentos por campo ou por linha de descrição de campo, contendo metadados mais ricos como `{"registro_id": "C170", "campo_nome": "COD_ITEM", "descricao_campo": "Código do item", "tipo": "C", "tam": "060"}`.

2.  **Ajustar Metadados e Indexação:**
    *   **Ação:** Atualizar a etapa de criação de `Document`s para incluir os metadados detalhados extraídos no passo anterior. Reconstruir o índice FAISS com esses documentos mais granulares.

3.  **Refinar o Prompt do LLM:**
    *   **Desafio:** O prompt padrão da cadeia `stuff` não instrui o LLM a usar os metadados para citação detalhada.
    *   **Ação:** Criar um `PromptTemplate` customizado para a `RetrievalQA` chain. O novo prompt deve instruir explicitamente o LLM a:
        *   Responder à pergunta.
        *   Utilizar os metadados (`registro_id`, `campo_nome`, etc.) dos documentos recuperados (`context`) para indicar claramente onde (em qual registro e campo) a informação base da resposta foi encontrada.

4.  **Avaliação e Iteração:**
    *   **Ação:** Testar o sistema RAG refinado com diversas perguntas, focando em verificar se a citação da fonte (registro e campo) está correta e útil.
    *   **Iteração:** Ajustar a extração de estrutura, os metadados e o prompt conforme necessário, com base nos resultados da avaliação.

5.  **(Opcional) Exploração Adicional:**
    *   Se o FLAN-T5-small local tiver dificuldade com as instruções complexas de citação, experimentar modelos maiores (localmente ou via API, se viável).
    *   Testar outros tipos de cadeia (`map_reduce`, `refine`) se a quantidade de informação recuperada ou o prompt se tornarem muito longos para o `stuff`.
    *   Considerar técnicas de extração de informação mais avançadas se a estrutura dos PDFs for muito complexa.

