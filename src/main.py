import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from processar_pdfs import processar_pdfs
import logging
from analise_textos import AnalisadorTextos
import matplotlib.pyplot as plt
import seaborn as sns


def log_vetores_e_vocabulario(vectorizer, X, textos, caminho_arquivo="vocabulario_e_vetores.txt"):
    """Exibe e salva o vocabulário e os vetores de frequência para cada texto."""
    try:
        vocabulario = vectorizer.get_feature_names_out()
        X_array = X.toarray()

        with open(caminho_arquivo, 'w', encoding='utf-8') as f:
            f.write("=== Análise de Vocabulário e Vetores ===\n")
            f.write(f"Total de termos no vocabulário: {len(vocabulario)}\n\n")

            # amostra do vocabulário (primeiros 100 termos)
            f.write("Amostra do Vocabulário:\n")
            f.write(", ".join(vocabulario[:100]) + "...\n\n")

            # estatísticas por texto
            f.write("=== Estatísticas por Texto ===\n")
            for i, (texto, vetor) in enumerate(zip(textos, X_array)):
                palavras_relevantes = [(vocabulario[idx], freq)
                                       for idx, freq in enumerate(vetor)
                                       if freq > 0]
                palavras_relevantes.sort(key=lambda x: x[1], reverse=True)

                f.write(f"\nTexto {i + 1}:\n")
                f.write(
                    f"Número de termos relevantes: {len(palavras_relevantes)}\n")
                f.write("Top 10 termos mais relevantes:\n")
                for palavra, freq in palavras_relevantes[:10]:
                    f.write(f"  {palavra}: {freq:.4f}\n")

        logging.info(f"Análise de vocabulário salva em: {caminho_arquivo}")
    except Exception as e:
        logging.error(f"Erro ao gerar log de vocabulário: {str(e)}")


def main():
   # logging para depuração

    logging.basicConfig(
        filename='processamento_completo.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # buscando o diretório raiz do projeto
        diretorio_raiz = os.path.dirname(os.path.abspath(__file__))

        # subdiretórios de pdfs
        diretorios_pdfs = {
            "horror": os.path.join(diretorio_raiz, "pdfsHorror"),
            "poetry": os.path.join(diretorio_raiz, "pdfsPoetry"),
            "romance": os.path.join(diretorio_raiz, "pdfsRomance")
        }

        # processando pdfs
        logging.info("Iniciando processamento dos PDFs...")
        textos, classes = processar_pdfs(diretorio_raiz, diretorios_pdfs)
        logging.info(
            f"Processamento concluído. Total de textos: {len(textos)}")

        # criando matriz TF-IDF
        logging.info("Criando matriz TF-IDF...")
        vectorizer = TfidfVectorizer(
            min_df=2,  # ignora termos que aparecem em menos de 2 documentos
            max_df=0.95,  # ignora termos que aparecem em mais de 95% dos documentos
            stop_words='english'
        )
        X = vectorizer.fit_transform(textos)
        logging.info(f"Matriz TF-IDF criada. Dimensões: {X.shape}")

        # gerando logs
        log_vetores_e_vocabulario(vectorizer, X, textos)

        # divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, classes, test_size=0.3, random_state=42, stratify=classes
        )
        logging.info("Divisão treino/teste realizada com sucesso")

        #  informações sobre a divisão
        with open('divisao_dados.log', 'w') as f:
            f.write(f"Total de amostras: {len(classes)}\n")
            f.write(f"Amostras de treino: {len(y_train)}\n")
            f.write(f"Amostras de teste: {len(y_test)}\n")

        return X_train, X_test, y_train, y_test, vectorizer.get_feature_names_out()

    except Exception as e:
        logging.error(f"Erro durante a execução: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, vocabulario = main()
        logging.info("Pipeline concluído com sucesso!")
    except Exception as e:
        logging.error(f"Erro na execução do script: {str(e)}")
