import os
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Baixar recursos do NLTK (apenas na primeira execução)
nltk.download('punkt')
nltk.download('stopwords')

def pdf_para_txt(caminho_pdf):
    with open(caminho_pdf, 'rb') as f:
        leitor = PyPDF2.PdfReader(f)
        texto = ""
        for pagina in range(len(leitor.pages)):
            texto += leitor.pages[pagina].extract_text() or ""
    return texto

def limpar_texto(texto):
    stop_words = set(stopwords.words('english'))
    palavras = word_tokenize(texto.lower())
    palavras_limpa = [palavra for palavra in palavras if palavra.isalnum() and palavra not in stop_words]
    return " ".join(palavras_limpa)

def salvar_texto_em_arquivo(nome_arquivo, texto_limpo, diretorio_raiz):
    diretorio_saida = os.path.join(diretorio_raiz, "textos_extraidos")
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_arquivo = os.path.join(diretorio_saida, f"{nome_arquivo}.txt")
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write(texto_limpo)

def processar_pdfs(diretorio_raiz, diretorios_pdfs):
    textos = []
    classes = []

    for classe, caminho_pdfs in diretorios_pdfs.items():
        if not os.path.exists(caminho_pdfs):
            print(f"Diretório {caminho_pdfs} não encontrado. Ignorando...")
            continue

        for arquivo in os.listdir(caminho_pdfs):
            if arquivo.endswith('.pdf'):
                caminho_arquivo_pdf = os.path.join(caminho_pdfs, arquivo)
                print(f"Processando: {arquivo}")
                texto_extraido = pdf_para_txt(caminho_arquivo_pdf)
                if texto_extraido.strip():
                    texto_limpo = limpar_texto(texto_extraido)
                    textos.append(texto_limpo)
                    classes.append(classe)
                    nome_arquivo_txt = os.path.splitext(arquivo)[0]
                    salvar_texto_em_arquivo(nome_arquivo_txt, texto_limpo, diretorio_raiz)
                    print(f"Texto salvo em: textos_extraidos/{nome_arquivo_txt}.txt")
                else:
                    print(f"O arquivo {arquivo} está vazio. Ignorando...")

    if not textos:
        raise ValueError("Nenhum texto foi extraído. Verifique os PDFs e diretórios.")

    return textos, classes
