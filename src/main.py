import os
import json
import logging
import cProfile
import pstats
from collections import Counter
import nltk

from otimizador import OtimizadorModelos
from analise import AnalisadorTextos, salvar_metricas_distribuicao
from vetorizacao import ProcessadorVetorial, verificar_distribuicao
from processamento import processar_pdfs, criar_diretorios_saida
from classificador import ClassificadorGeneros

logging.basicConfig(
    filename='processamento.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main_process():
    """Função principal para executar todas as análises com profiling."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
        diretorios = criar_diretorios_saida(diretorio_raiz)

        # Definindo os diretórios de PDFs por gênero
        diretorios_pdfs = {
            "horror": os.path.join(diretorio_raiz, "pdfsHorror"),
            "poetry": os.path.join(diretorio_raiz, "pdfsPoetry"),
            "romance": os.path.join(diretorio_raiz, "pdfsRomance")
        }

        logging.info("Iniciando processamento dos PDFs...")
        textos, textos_originais, classes = processar_pdfs(
            diretorio_raiz, diretorios_pdfs)
        logging.info(
            f"Processamento concluído. Total de textos: {len(textos)}")

        # Verificar número mínimo de amostras por classe
        contagem_classes = Counter(classes)
        if min(contagem_classes.values()) < 2:
            raise ValueError(
                f"Número insuficiente de amostras por classe: {dict(contagem_classes)}")

        analisador = AnalisadorTextos()
        analisador.analisar_distribuicao_tamanhos(textos, classes)
        vocab_relevante = analisador.analisar_vocabulario(textos, min_freq=5)
        analisador.gerar_wordcloud(vocab_relevante)
        analisador.analisar_caracteristicas_distintas(textos, classes)

        logging.info("Iniciando vetorização e divisão dos dados...")
        processador = ProcessadorVetorial(
            min_df=3,              # Frequência mínima dos termos
            max_df=0.95,           # Frequência máxima (%)
            ngram_range=(1, 2),    # Uni e bigramas
            max_features=50000,    # Tamanho máximo do vocabulário ajustado
            balanceamento='combinado'
        )

        # Realizando a vetorização e divisão
        X_train, X_test, y_train, y_test = processador.vetorizar_e_dividir(
            textos=textos,
            classes=classes,
            test_size=0.3,         # 30% para teste
            random_state=42        # Semente para reprodutibilidade
        )

        verificar_distribuicao(y_train)

        # Gerando visualizações da vetorização
        processador.visualizar_importancia_termos()

        # Salvando métricas de distribuição
        salvar_metricas_distribuicao(
            textos, classes, diretorios['graficos'])

        # Treinamento e avaliação dos classificadores
        logging.info("Iniciando treinamento dos classificadores...")
        classificador = ClassificadorGeneros(X_train, y_train, X_test, y_test)
        classificador.configurar_classificadores()
        classificador.treinar_e_avaliar()
        classificador.salvar_resultados(diretorios['resultados'])

        # Salvando informações sobre a divisão dos dados
        with open(os.path.join(diretorios['logs'], 'divisao_dados.log'), 'w') as f:
            f.write("=== Informações sobre a Divisão dos Dados ===\n")
            f.write(f"Total de documentos: {len(classes)}\n")
            f.write(f"Documentos no conjunto de treino: {len(y_train)}\n")
            f.write(f"Documentos no conjunto de teste: {len(y_test)}\n")
            f.write(f"\nDimensões da matriz de treino: {X_train.shape}\n")
            f.write(f"Dimensões da matriz de teste: {X_test.shape}\n")

        # Otimização dos modelos
        logging.info("Iniciando otimização dos modelos...")
        otimizador = OtimizadorModelos(
            X_train, y_train, X_test, y_test, desired_folds=5)
        otimizador.otimizar_todos_modelos()
        otimizador.gerar_graficos_comparativos(
            diretorio_saida=diretorios['resultados'])
        otimizador.salvar_resultados(diretorio_saida=diretorios['resultados'])
        resultados_significancia = otimizador.avaliar_significancia()

        # Salvando resultados de significância
        with open(os.path.join(diretorios['resultados'], 'resultados_significancia.json'), 'w') as f:
            json.dump(resultados_significancia, f, indent=4)

        logging.info("Pipeline completo executado com sucesso!")

    except Exception as e:
        logging.error(f"Erro durante a execução: {str(e)}")
        raise

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
        stats_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'profiling.prof')
        stats.dump_stats(stats_file)
        logging.info(f"Profiling salvo em {stats_file}")


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.environ['NLTK_DATA'] = nltk_data_dir
    try:
        main_process()
    except Exception as e:
        logging.error(f"Erro na execução do script: {str(e)}")
