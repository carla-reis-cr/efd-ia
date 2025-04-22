@echo off
setlocal

REM --- Configuração ---
set ENV_NAME=efd-ia
set ENV_FILE=environment.yml
set PYTHON_SCRIPT_ENV=config\env.py
REM set PYTHON_SCRIPT_SETUP=config\setup_nltk.py (Removido)

REM --- Verificações ---
echo Verificando existencia do Conda...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Erro: Conda nao encontrado no PATH.
    echo Certifique-se de que Anaconda ou Miniconda estao instalados e configurados corretamente.
    goto :eof
)

if not exist "%ENV_FILE%" (
    echo Erro: Arquivo de ambiente '%ENV_FILE%' nao encontrado.
    goto :eof
)
if not exist "%PYTHON_SCRIPT_ENV%" (
    echo Erro: Script de ambiente '%PYTHON_SCRIPT_ENV%' nao encontrado.
    goto :eof
)
REM Verificação do script de setup removida

REM --- Criação do Ambiente ---
echo Criando o ambiente Conda '%ENV_NAME%' a partir de '%ENV_FILE%'...
echo Isso pode levar varios minutos.
conda env create -f "%ENV_FILE%" --force
if %errorlevel% neq 0 (
    echo Erro: Falha ao criar o ambiente Conda '%ENV_NAME%'.
    goto :eof
)
echo Ambiente '%ENV_NAME%' criado com sucesso.

REM --- Execução dos Scripts de Configuração ---
echo Ativando ambiente '%ENV_NAME%' e executando scripts de configuracao...

REM A ativação dentro de um script batch é um pouco mais complexa.
REM Usamos 'conda run' que executa um comando dentro do ambiente especificado.

echo Executando %PYTHON_SCRIPT_ENV%...
conda run -n %ENV_NAME% python "%PYTHON_SCRIPT_ENV%"
if %errorlevel% neq 0 (
    echo Erro: Falha ao executar '%PYTHON_SCRIPT_ENV%' no ambiente '%ENV_NAME%'.
    goto :eof
)

REM Execução do script de setup removida

echo.
echo ------------------------------------------------------------------
echo Processo de criacao e configuracao do ambiente concluido!
echo Para ativar o ambiente manualmente, use: conda activate %ENV_NAME%
echo Nota: Dados NLTK serao baixados na primeira utilizacao.
echo ------------------------------------------------------------------

endlocal 