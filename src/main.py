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


def criar_diretorios_saida(diretorio_raiz):
    """Cria diretórios necessários para salvar as análises."""
    diretorios = ['analises', 'graficos', 'logs']
    for dir_nome in diretorios:
        caminho = os.path.join(diretorio_raiz, dir_nome)
        os.makedirs(caminho, exist_ok=True)
    return {nome: os.path.join(diretorio_raiz, nome) for nome in diretorios}


def salvar_metricas_distribuicao(textos, classes, dir_graficos):
    """Salva gráficos e métricas sobre a distribuição dos textos."""
    # retorna a distribuição do tamanho dos textos por classe
    tamanhos = [len(texto.split()) for texto in textos]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=classes, y=tamanhos)
    plt.title('Distribuição do Tamanho dos Textos por Gênero')
    plt.xlabel('Gênero')
    plt.ylabel('Número de Palavras')
    plt.savefig(os.path.join(dir_graficos, 'distribuicao_tamanhos.png'))
    plt.close()


def main():
    logging.basicConfig(
        filename='processamento_completo.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # buscando o diretório raiz do projeto
        diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
        diretorios = criar_diretorios_saida(diretorio_raiz)

        # subdiretórios de PDFs
        diretorios_pdfs = {
            "horror": os.path.join(diretorio_raiz, "pdfsHorror"),
            "poetry": os.path.join(diretorio_raiz, "pdfsPoetry"),
            "romance": os.path.join(diretorio_raiz, "pdfsRomance")
        }

        # processando PDFs
        logging.info("Iniciando processamento dos PDFs...")
        textos, textos_originais, classes = processar_pdfs(
            diretorio_raiz, diretorios_pdfs)
        logging.info(
            f"Processamento concluído. Total de textos: {len(textos)}")

        # retornando análise dos textos
        analisador = AnalisadorTextos()
        analisador.analisar_distribuicao_tamanhos(
            textos, classes)  # Salva gráficos
        vocab_relevante = analisador.analisar_vocabulario(textos, min_freq=5)
        analisador.analisar_caracteristicas_distintas(textos, classes)

        # validando a limpeza
        for i, (orig, limpo) in enumerate(zip(textos_originais, textos)):
            if not analisador.validar_limpeza(orig, limpo):
                logging.warning(
                    f"Possível limpeza excessiva detectada no texto {i+1}")

        # criando matriz TF-IDF
        logging.info("Criando matriz TF-IDF...")
        vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        X = vectorizer.fit_transform(textos)
        logging.info(f"Matriz TF-IDF criada. Dimensões: {X.shape}")

        # salvando análises e logs
        log_vetores_e_vocabulario(vectorizer, X, textos, os.path.join(
            diretorios['analises'], 'vocabulario_e_vetores.txt'))
        salvar_metricas_distribuicao(
            textos, classes, diretorios['graficos'])

        # divindo os dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, classes, test_size=0.3, random_state=42, stratify=classes
        )
        logging.info("Divisão treino/teste realizada com sucesso")

        # informações sobre a divisão
        with open(os.path.join(diretorios['logs'], 'divisao_dados.log'), 'w') as f:
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
