import os
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import logging
import shutil
from langdetect import detect
import chardet
from typing import List, Tuple, Dict
import pandas as pd

# logging (para depuração)
logging.basicConfig(
    filename='processamento_pdfs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# baixando recursos do NLTK
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Erro ao baixar recursos NLTK: {str(e)}")


def criar_backup(diretorio_pdfs):
    """Cria backup dos arquivos originais."""
    backup_dir = diretorio_pdfs + "_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(diretorio_pdfs, backup_dir)
        logging.info(f"Backup criado em: {backup_dir}")
    return backup_dir


def detectar_encoding(texto_bytes):
    """Detecta o encoding do texto."""
    resultado = chardet.detect(texto_bytes)
    return resultado['encoding']


def verificar_lingua(texto, lingua_esperada='en'):
    """Verifica se o texto está na língua esperada."""
    try:
        return detect(texto) == lingua_esperada
    except:
        return False


def pdf_para_txt(caminho_pdf):
    """Extrai texto do PDF com verificações adicionais."""
    texto = ""
    try:
        with open(caminho_pdf, 'rb') as f:
            leitor = PyPDF2.PdfReader(f)

            # Verifica se o PDF está corrompido
            if not leitor.pages:
                logging.warning(f"PDF possivelmente corrompido: {caminho_pdf}")
                return ""

            for pagina in leitor.pages:
                conteudo = pagina.extract_text() or ""

                # verificando se a página contém principalmente imagens
                if len(conteudo.strip()) < 50:  # retorno - página com pouco texto com pouco texto
                    logging.warning(
                        f"Página possivelmente contém principalmente imagens: {caminho_pdf}")

                texto += conteudo

    except Exception as e:
        logging.error(f"Erro ao processar PDF {caminho_pdf}: {str(e)}")
        return ""

    return texto


def limpar_texto(texto):
    """Limpa e normaliza o texto."""
    try:
        # remove caracteres especiai, mantém apenas alfanuméricos e espaços
        texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)

        # convertendo o texto para minúsculas
        texto = texto.lower()

        # removendo múltiplos espaços
        texto = re.sub(r'\s+', ' ', texto)

        # removendo stopwords
        stop_words = set(stopwords.words('english'))
        palavras = word_tokenize(texto)
        palavras_limpa = [palavra for palavra in palavras if palavra.isalnum(
        ) and palavra not in stop_words]

        return " ".join(palavras_limpa)
    except Exception as e:
        logging.error(f"Erro na limpeza do texto: {str(e)}")
        return ""


def salvar_texto_em_arquivo(nome_arquivo, texto_limpo, diretorio_raiz, classe):
    """Salva o texto processado com encoding UTF-8."""
    try:
        diretorio_saida = os.path.join(
            diretorio_raiz, "textos_extraidos", classe)
        os.makedirs(diretorio_saida, exist_ok=True)

        caminho_arquivo = os.path.join(diretorio_saida, f"{nome_arquivo}.txt")
        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write(texto_limpo)

        logging.info(f"Texto salvo com sucesso: {caminho_arquivo}")
        return True
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo {nome_arquivo}: {str(e)}")
        return False


def validar_texto(texto):
    """Realiza validações no texto extraído."""
    if not texto.strip():
        return False, "Texto vazio"

    if len(texto.split()) < 100:  # Texto muito curto
        return False, "Texto muito curto"

    if not verificar_lingua(texto):
        return False, "Idioma incorreto"

    return True, "OK"


def processar_pdfs(diretorio_raiz, diretorios_pdfs):
    """Processa os PDFs com validações e logging."""
    textos = []
    classes = []
    estatisticas = {
        'total_processado': 0,
        'sucessos': 0,
        'falhas': 0,
        'por_classe': {}
    }

    for classe, caminho_pdfs in diretorios_pdfs.items():
        if not os.path.exists(caminho_pdfs):
            logging.error(f"Diretório não encontrado: {caminho_pdfs}")
            continue

        # criando backup antes de processar
        backup_dir = criar_backup(caminho_pdfs)
        estatisticas['por_classe'][classe] = {'processados': 0, 'falhas': 0}

        for arquivo in os.listdir(caminho_pdfs):
            if not arquivo.endswith('.pdf'):
                continue

            estatisticas['total_processado'] += 1
            caminho_arquivo_pdf = os.path.join(caminho_pdfs, arquivo)
            logging.info(f"Processando: {arquivo} (Classe: {classe})")

            # extraindo o texto
            texto_extraido = pdf_para_txt(caminho_arquivo_pdf)

            #  validando o texto
            valido, motivo = validar_texto(texto_extraido)
            if not valido:
                logging.warning(f"Texto inválido em {arquivo}: {motivo}")
                estatisticas['falhas'] += 1
                estatisticas['por_classe'][classe]['falhas'] += 1
                continue

            # limpando e normalizando
            texto_limpo = limpar_texto(texto_extraido)
            if texto_limpo:
                nome_arquivo_txt = os.path.splitext(arquivo)[0]
                if salvar_texto_em_arquivo(nome_arquivo_txt, texto_limpo, diretorio_raiz, classe):
                    textos.append(texto_limpo)
                    classes.append(classe)
                    estatisticas['sucessos'] += 1
                    estatisticas['por_classe'][classe]['processados'] += 1

    # log final das estatísticas
    logging.info("=== Estatísticas de Processamento ===")
    logging.info(f"Total processado: {estatisticas['total_processado']}")
    logging.info(f"Sucessos: {estatisticas['sucessos']}")
    logging.info(f"Falhas: {estatisticas['falhas']}")
    for classe, stats in estatisticas['por_classe'].items():
        logging.info(f"Classe {classe}: Processados {
                     stats['processados']}, Falhas {stats['falhas']}")

    if not textos:
        raise ValueError("Nenhum texto foi extraído com sucesso.")

    return textos, classes
