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

# baixando recursos da biblioteca NLTK
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Erro ao baixar recursos NLTK: {str(e)}")


def criar_backup(diretorio_pdfs):
    """Cria backup dos arquivos originais."""
    backup_dir = diretorio_pdfs + "_backup"
    try:
        if not os.path.exists(backup_dir):
            shutil.copytree(diretorio_pdfs, backup_dir)
            logging.info(f"Backup criado em: {backup_dir}")
    except Exception as e:
        logging.error(f"Erro ao criar backup: {str(e)}")
    return backup_dir


def detectar_encoding(texto_bytes):
    """Detecta o encoding do texto."""
    resultado = chardet.detect(texto_bytes)
    return resultado['encoding']


def verificar_lingua(texto, lingua_esperada='en'):
    """Verifica se o texto está na língua esperada."""
    try:
        return detect(texto) == lingua_esperada
    except Exception as e:
        logging.error(f"Erro ao detectar língua: {str(e)}")
        return False


def pdf_para_txt(caminho_pdf):
    """Extrai texto do PDF com verificações adicionais."""
    texto = ""
    try:
        with open(caminho_pdf, 'rb') as f:
            leitor = PyPDF2.PdfReader(f)

            # verifica se o PDF está corrompido
            if not leitor.pages:
                logging.warning(f"PDF possivelmente corrompido: {caminho_pdf}")
                return ""

            # verificando se a página contém principalmente imagens
            for pagina in leitor.pages:
                conteudo = pagina.extract_text() or ""

                if len(conteudo.strip()) < 50:  # retorno - página com pouco texto
                    logging.warning(
                        f"Página possivelmente contém principalmente imagens: {caminho_pdf}")

                texto += conteudo

    except Exception as e:
        logging.error(f"Erro ao processar PDF {caminho_pdf}: {str(e)}")
        return ""

    return texto


def limpar_texto(texto: str, preservar_palavras: set = None) -> Tuple[str, str]:
    """
    Limpa e normaliza o texto, preservando palavras importantes.

    Args:
        texto: Texto a ser limpo
        preservar_palavras: Conjunto de palavras a serem preservadas mesmo que sejam stopwords
    """
    try:
        # guardando o texto original para análise
        texto_original = texto

        # removendo caracteres especiais mas preservando alguns importantes para estilo
        texto = re.sub(r'[^a-zA-Z0-9\s\-\'"]', ' ', texto)

        # convertendo para minúsculas
        texto = texto.lower()

        # removendo múltiplos espaços
        texto = re.sub(r'\s+', ' ', texto)

        # tokenizando
        palavras = word_tokenize(texto)

        # removendo stopwords, mas preservando palavras importantes
        stop_words = set(stopwords.words('english'))
        if preservar_palavras:
            stop_words = stop_words - preservar_palavras

        palavras_limpa = []
        for palavra in palavras:
            # mantenho palavras que não são stopwords ou que devem ser preservadas
            if (palavra not in stop_words) or (preservar_palavras and palavra in preservar_palavras):
                palavras_limpa.append(palavra)

        texto_limpo = " ".join(palavras_limpa)

        # registrando estatísticas de limpeza
        stats = {
            'palavras_originais': len(texto_original.split()),
            'palavras_apos_limpeza': len(texto_limpo.split()),
            'proporcao_mantida': len(texto_limpo.split()) /
            len(texto_original.split()) if texto_original else 0
        }

        logging.info(f"Estatísticas de limpeza: {stats}")

        return texto_limpo, texto_original

    except Exception as e:
        logging.error(f"Erro na limpeza do texto: {str(e)}")
        return "", texto


def salvar_texto_em_arquivo(nome_arquivo: str, texto_limpo: str,
                            texto_original: str, diretorio_raiz: str,
                            classe: str) -> bool:
    """Salva tanto o texto limpo quanto o original."""
    try:
        # diretório para textos limpos
        diretorio_saida_limpo = os.path.join(
            diretorio_raiz, "textos_extraidos", classe, "limpos")
        # diretório para textos originais
        diretorio_saida_original = os.path.join(
            diretorio_raiz, "textos_extraidos", classe, "originais")

        os.makedirs(diretorio_saida_limpo, exist_ok=True)
        os.makedirs(diretorio_saida_original, exist_ok=True)

        # salvando texto limpo
        caminho_arquivo_limpo = os.path.join(
            diretorio_saida_limpo, f"{nome_arquivo}.txt")
        with open(caminho_arquivo_limpo, 'w', encoding='utf-8') as f:
            f.write(texto_limpo)

        # salvando texto original
        caminho_arquivo_original = os.path.join(
            diretorio_saida_original, f"{nome_arquivo}_original.txt")
        with open(caminho_arquivo_original, 'w', encoding='utf-8') as f:
            f.write(texto_original)

        logging.info(f"textos salvos com sucesso: {caminho_arquivo_limpo}")
        return True
    except Exception as e:
        logging.error(f"erro ao salvar arquivo {nome_arquivo}: {str(e)}")
        return False


def validar_texto(texto: str) -> Tuple[bool, str]:
    """Realiza validações no texto extraído."""
    if not texto.strip():
        return False, "Texto vazio"

    if len(texto.split()) < 100:
        return False, "Texto muito curto"

    if not verificar_lingua(texto):
        return False, "Idioma incorreto"

    return True, "OK"


def processar_pdfs(diretorio_raiz: str,
                   diretorios_pdfs: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Processa os PDFs e retorna textos limpos, originais e classes."""
    textos_limpos = []
    textos_originais = []
    classes = []
    estatisticas = {
        'total_processado': 0,
        'sucessos': 0,
        'falhas': 0,
        'por_classe': {}
    }

    # palavras importantes a serem preservadas por gênero, devem ser mantidas - definidas em dupla
    palavras_preservar = {
        'horror': {'fear', 'dark', 'blood', 'death', 'night', 'ghost', 'shadow'},
        'poetry': {'love', 'heart', 'soul', 'dream', 'light', 'sky', 'wind'},
        'romance': {'love', 'heart', 'kiss', 'smile', 'eyes', 'touch', 'feel'}
    }

    for classe, caminho_pdfs in diretorios_pdfs.items():
        if not os.path.exists(caminho_pdfs):
            logging.error(f"Diretório não encontrado: {caminho_pdfs}")
            continue

        # criando backup antes de processar
        backup_dir = criar_backup(caminho_pdfs)
        estatisticas['por_classe'][classe] = {'processados': 0, 'falhas': 0}

        # obtendo palavras a preservar para este gênero
        preservar = palavras_preservar.get(classe, set())

        for arquivo in os.listdir(caminho_pdfs):
            if not arquivo.endswith('.pdf'):
                continue

            estatisticas['total_processado'] += 1
            caminho_arquivo_pdf = os.path.join(caminho_pdfs, arquivo)
            logging.info(f"Processando: {arquivo} (Classe: {classe})")

            # extraindo o texto
            texto_extraido = pdf_para_txt(caminho_arquivo_pdf)

            # validando o texto
            valido, motivo = validar_texto(texto_extraido)
            if not valido:
                logging.warning(f"Texto inválido em {arquivo}: {motivo}")
                estatisticas['falhas'] += 1
                estatisticas['por_classe'][classe]['falhas'] += 1
                continue

            # limpando e normalizando
            texto_limpo, texto_original = limpar_texto(
                texto_extraido, preservar)

            if texto_limpo:
                nome_arquivo_txt = os.path.splitext(arquivo)[0]
                if salvar_texto_em_arquivo(nome_arquivo_txt, texto_limpo,
                                           texto_original, diretorio_raiz, classe):
                    textos_limpos.append(texto_limpo)
                    textos_originais.append(texto_original)
                    classes.append(classe)
                    estatisticas['sucessos'] += 1
                    estatisticas['por_classe'][classe]['processados'] += 1

            # salvando estatísticas detalhadas do texto
            stats_texto = {
                'arquivo': arquivo,
                'classe': classe,
                'tamanho_original': len(texto_original.split()) if texto_original else 0,
                'tamanho_limpo': len(texto_limpo.split()) if texto_limpo else 0,
                'proporcao_mantida': len(texto_limpo.split()) / len(texto_original.split())
                if texto_original and texto_limpo else 0
            }

            # bloco desativado
            """
            salvando em um DataFrame para análise posterior, caso for preciso
            df_stats = pd.DataFrame([stats_texto])
            stats_file = os.path.join(
                diretorio_raiz, 'estatisticas_processamento.csv')
            df_stats.to_csv(stats_file, mode='a', header=not os.path.exists(stats_file),
                            index=False)
            """

    # log final das estatísticas
    logging.info("=== Estatísticas de Processamento ===")
    logging.info(f"Total processado: {estatisticas['total_processado']}")
    logging.info(f"Sucessos: {estatisticas['sucessos']}")
    logging.info(f"Falhas: {estatisticas['falhas']}")
    for classe, stats in estatisticas['por_classe'].items():
        logging.info(f"Classe {classe}: Processados {stats['processados']}, "
                     f"Falhas {stats['falhas']}")

    # salvando métricas gerais
    metricas_gerais = {
        'total_documentos': estatisticas['total_processado'],
        'documentos_processados': estatisticas['sucessos'],
        'documentos_falhos': estatisticas['falhas'],
        'taxa_sucesso': estatisticas['sucessos'] / estatisticas['total_processado']
        if estatisticas['total_processado'] > 0 else 0
    }

    df_metricas = pd.DataFrame([metricas_gerais])
    df_metricas.to_csv(os.path.join(
        diretorio_raiz, 'metricas_gerais.csv'), index=False)

    if not textos_limpos:
        raise ValueError("Nenhum texto foi extraído com sucesso.")

    return textos_limpos, textos_originais, classes
