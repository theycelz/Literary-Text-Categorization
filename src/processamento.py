import os
import logging
import multiprocessing
from collections import Counter
from typing import Dict, Tuple
import pandas as pd
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
import chardet
import re

STOP_WORDS = set(stopwords.words('english'))


def processar_pdf(args):
    arquivo, classe, preservar_palavras, diretorio_raiz = args
    try:
        logging.info("Iniciando processamento")
        texto_extraido = pdf_para_txt(arquivo)
        valido, motivo = validar_texto(texto_extraido)
        if not valido:
            logging.warning(f"Texto inválido em {arquivo}: {motivo}", extra={
                            'nome_funcao': 'processar_pdf'})
            return None
        texto_limpo, texto_original = limpar_texto(
            texto_extraido, preservar_palavras)
        if texto_limpo:
            nome_arquivo_txt = os.path.splitext(os.path.basename(arquivo))[0]
            if salvar_texto_em_arquivo(nome_arquivo_txt, texto_limpo, texto_original, diretorio_raiz, classe):
                return (texto_limpo, texto_original, classe)
    except Exception as e:
        logging.error(f"Erro: {arquivo} - {str(e)}")
    return None


def processar_pdfs(diretorio_raiz: str, diretorios_pdfs: Dict[str, str]):
    if not os.path.isdir(diretorio_raiz):
        raise ValueError(f"Diretório raiz inválido: {diretorio_raiz}")

    for classe, caminho in diretorios_pdfs.items():
        if not os.path.isdir(caminho):
            raise ValueError(
                f"Diretório inválido para classe {classe}: {caminho}")

    if not diretorios_pdfs:
        raise ValueError("Nenhum diretório de PDFs fornecido")
    textos_limpos = []
    textos_originais = []
    classes = []
    manager = multiprocessing.Manager()
    estatisticas = manager.dict({
        'total_processado': 0,
        'sucessos': 0,
        'falhas': 0,
        'por_classe': manager.dict({
            classe: manager.dict({'processados': 0, 'falhas': 0})
            for classe in diretorios_pdfs.keys()
        })
    })

    lock = multiprocessing.Lock()

    def atualizar_estatisticas(resultado, classe):
        with lock:
            if resultado:
                estatisticas['sucessos'] += 1
                estatisticas['por_classe'][classe]['processados'] += 1
            else:
                estatisticas['falhas'] += 1
                estatisticas['por_classe'][classe]['falhas'] += 1

    palavras_preservar_dict = {
        'horror': {'fear', 'dark', 'blood', 'death', 'night', 'ghost', 'shadow'},
        'poetry': {'love', 'heart', 'soul', 'dream', 'light', 'sky', 'wind'},
        'romance': {'love', 'heart', 'kiss', 'smile', 'eyes', 'touch', 'feel'}
    }

    tarefas = []
    for classe, caminho_pdfs in diretorios_pdfs.items():
        arquivos_pdf = [os.path.join(caminho_pdfs, f) for f in os.listdir(
            caminho_pdfs) if f.endswith('.pdf')]
        for arquivo_pdf in arquivos_pdf:
            tarefas.append((arquivo_pdf, classe, palavras_preservar_dict.get(
                classe, set()), diretorio_raiz))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        resultados = pool.map(processar_pdf, tarefas)

    for resultado in resultados:
        if resultado:
            texto_limpo, texto_original, classe = resultado
            textos_limpos.append(texto_limpo)
            textos_originais.append(texto_original)
            classes.append(classe)
            estatisticas['sucessos'] += 1
            estatisticas['por_classe'][classe]['processados'] += 1
        else:
            estatisticas['falhas'] += 1

    logging.info("=== Estatísticas de Processamento ===")
    logging.info(f"Total processado: {len(tarefas)}")
    logging.info(f"Sucessos: {estatisticas['sucessos']}")
    logging.info(f"Falhas: {estatisticas['falhas']}")
    for classe, stats in estatisticas['por_classe'].items():
        logging.info(
            f"Classe {classe}: Processados {stats['processados']}, Falhas {stats['falhas']}")

    metricas_gerais = {
        'total_documentos': len(tarefas),
        'documentos_processados': estatisticas['sucessos'],
        'documentos_falhos': estatisticas['falhas'],
        'taxa_sucesso': estatisticas['sucessos'] / len(tarefas) if len(tarefas) > 0 else 0
    }

    df_metricas = pd.DataFrame([metricas_gerais])
    df_metricas.to_csv(os.path.join(
        diretorio_raiz, 'metricas_gerais.csv'), index=False)

    if not textos_limpos:
        raise ValueError("Nenhum texto foi extraído com sucesso.")

    return textos_limpos, textos_originais, classes


def criar_diretorios_saida(diretorio_raiz):
    diretorios = ['analises', 'graficos',
                  'logs', 'resultados', 'textos_extraidos']
    for dir_nome in diretorios:
        caminho = os.path.join(diretorio_raiz, dir_nome)
        os.makedirs(caminho, exist_ok=True)
    return {nome: os.path.join(diretorio_raiz, nome) for nome in diretorios}


def pdf_para_txt(caminho_pdf):
    if not os.path.exists(caminho_pdf):
        logging.error(f"Arquivo não encontrado: {caminho_pdf}")
        return ""

    try:
        texto = extract_text(caminho_pdf)
        if not texto or not texto.strip():
            logging.warning(f"PDF vazio ou sem texto: {caminho_pdf}")
            return ""
        return texto.strip()
    except Exception as e:
        logging.error(f"Erro ao processar PDF {caminho_pdf}: {str(e)}")
        return ""


def limpar_texto(texto: str, preservar_palavras: set[str] = None) -> Tuple[str, str]:
    if not texto or not isinstance(texto, str):
        logging.error(f"Texto inválido ou vazio: {type(texto)}")
        return "", ""

    try:
        texto_original = texto
        texto = texto.lower()
        texto = texto.encode('ascii', errors='ignore').decode()
        texto = re.sub(r'[^a-z0-9\s\-\'"]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        texto = texto.replace("'", "")

        palavras = []
        try:
            palavras = word_tokenize(texto)
        except Exception as e:
            logging.warning(
                f"Erro na tokenização, usando split simples: {str(e)}")
            palavras = [palavra.strip()
                        for palavra in texto.split() if palavra.strip()]

        stop_words = STOP_WORDS.copy()
        if preservar_palavras and isinstance(preservar_palavras, set):
            stop_words -= set(palavra.lower()
                              for palavra in preservar_palavras)

        palavras_limpa = [
            palavra for palavra in palavras
            if (2 <= len(palavra) <= 45 and
                re.match(r'^[a-z\-\']+$', palavra) and
                (palavra not in stop_words or
                 (preservar_palavras and palavra in preservar_palavras)))
        ]

        texto_limpo = " ".join(palavras_limpa)

        if not texto_limpo:
            logging.warning("Texto ficou vazio após limpeza")
            return texto_original, texto_original

        palavras_originais = len(
            [p for p in texto_original.split() if p.strip()])
        palavras_final = len(palavras_limpa)
        proporcao = palavras_final / palavras_originais if palavras_originais > 0 else 0

        stats = {
            'palavras_originais': palavras_originais,
            'palavras_apos_limpeza': palavras_final,
            'proporcao_mantida': proporcao
        }

        if proporcao < 0.1:
            logging.warning(
                f"Limpeza muito agressiva: manteve apenas {proporcao*100:.1f}% das palavras")

        logging.info(f"Estatísticas de limpeza: {stats}")

        return texto_limpo, texto_original

    except Exception as e:
        logging.error(f"Erro na limpeza do texto: {str(e)}")
        return texto if isinstance(texto, str) else "", texto if isinstance(texto, str) else ""


def salvar_texto_em_arquivo(nome_arquivo: str, texto_limpo: str, texto_original: str, diretorio_raiz: str, classe: str) -> bool:
    try:
        diretorio_saida_limpo = os.path.join(
            diretorio_raiz, "textos_extraidos", classe, "limpos")
        diretorio_saida_original = os.path.join(
            diretorio_raiz, "textos_extraidos", classe, "originais")

        os.makedirs(diretorio_saida_limpo, exist_ok=True)
        os.makedirs(diretorio_saida_original, exist_ok=True)

        with open(os.path.join(diretorio_saida_limpo, f"{nome_arquivo}.txt"), 'w', encoding='utf-8') as f:
            f.write(texto_limpo)

        with open(os.path.join(diretorio_saida_original, f"{nome_arquivo}_original.txt"), 'w', encoding='utf-8') as f:
            f.write(texto_original)

        return True
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo {nome_arquivo}: {str(e)}")
        return False


def validar_texto(texto: str) -> Tuple[bool, str]:
    if not texto.strip():
        return False, "Texto vazio"

    if len(texto.split()) < 100:
        return False, "Texto muito curto"

    if not verificar_lingua(texto):
        return False, "Idioma incorreto"

    return True, "OK"


def verificar_lingua(texto, lingua_esperada='en'):
    try:
        return detect(texto) == lingua_esperada
    except Exception as e:
        logging.error(f"Erro ao detectar língua: {str(e)}")
        return False
