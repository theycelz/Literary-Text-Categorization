import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from processar_pdfs import processar_pdfs
import logging
from analise_textos import AnalisadorTextos
import matplotlib.pyplot as plt
import seaborn as sns
from vetorizacao import ProcessadorVetorial
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any


class ClassificadorGeneros:
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Inicializa o classificador com os dados de treino e teste.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # definindo métricas para validação cruzada
        self.scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }

        # dicionário para armazenar resultados
        self.resultados = {}

    def configurar_classificadores(self):
        """
        Configura os classificadores com hiperparâmetros iniciais conservadores
        para evitar overfitting.
        """
        self.classificadores = {
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='cosine'
            ),
            'NaiveBayes': MultinomialNB(
                alpha=1.0
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }

    def treinar_e_avaliar(self):
        """
        Treina e avalia cada classificador usando validação cruzada.
        """
        for nome, clf in self.classificadores.items():
            logging.info(f"Iniciando treinamento do classificador: {nome}")

            try:
                cv_results = cross_validate(
                    clf,
                    self.X_train,
                    self.y_train,
                    cv=5,
                    scoring=self.scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                metricas = {
                    metric: {
                        'media': cv_results[f'test_{metric}'].mean(),
                        'desvio_padrao': cv_results[f'test_{metric}'].std()
                    }
                    for metric in self.scoring.keys()
                }

                self.resultados[nome] = {
                    'metricas': metricas,
                    'tempo_treino': cv_results['fit_time'].mean()
                }

                logging.info(f"Treinamento concluído para {nome}")

            except Exception as e:
                logging.error(f"Erro no treinamento do classificador {
                              nome}: {str(e)}")
                continue

    def salvar_resultados(self, diretorio_saida: str = 'resultados'):
        """
        Salva os resultados em formato JSON e CSV.
        """
        os.makedirs(diretorio_saida, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # salvando em  um json
        caminho_json = os.path.join(
            diretorio_saida, f'resultados_{timestamp}.json')
        with open(caminho_json, 'w') as f:
            json.dump(self.resultados, f, indent=4)
        # convertendo para dataFrame e salvando em  um csv
        rows = []
        for clf_name, results in self.resultados.items():
            row = {'classificador': clf_name}
            for metric, values in results['metricas'].items():
                row[f'{metric}_media'] = values['media']
                row[f'{metric}_std'] = values['desvio_padrao']
            row['tempo_treino'] = results['tempo_treino']
            rows.append(row)

        df_resultados = pd.DataFrame(rows)
        caminho_csv = os.path.join(
            diretorio_saida, f'resultados_{timestamp}.csv')
        df_resultados.to_csv(caminho_csv, index=False)

        logging.info(f"Resultados salvos em {diretorio_saida}")


def criar_diretorios_saida(diretorio_raiz):
    """Cria diretórios necessários para salvar as análises."""
    diretorios = ['analises', 'graficos', 'logs', 'resultados']
    for dir_nome in diretorios:
        caminho = os.path.join(diretorio_raiz, dir_nome)
        os.makedirs(caminho, exist_ok=True)
    return {nome: os.path.join(diretorio_raiz, nome) for nome in diretorios}


def salvar_metricas_distribuicao(textos, classes, dir_graficos):
    """Salva gráficos e métricas sobre a distribuição dos textos."""

    # calculando a distribuição do tamanho dos textos por classe
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
        # 1. adicionando configuração inicial de diretórios
        diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
        diretorios = criar_diretorios_saida(diretorio_raiz)

        # 2. definindo os diretórios de PDFs por gênero
        diretorios_pdfs = {
            "horror": os.path.join(diretorio_raiz, "pdfsHorror"),
            "poetry": os.path.join(diretorio_raiz, "pdfsPoetry"),
            "romance": os.path.join(diretorio_raiz, "pdfsRomance")
        }

        # 3. processamento inicial dos PDFs
        logging.info("Iniciando processamento dos PDFs...")
        textos, textos_originais, classes = processar_pdfs(
            diretorio_raiz, diretorios_pdfs)
        logging.info(
            f"Processamento concluído. Total de textos: {len(textos)}")

        # 4. analise dos textos
        analisador = AnalisadorTextos()
        analisador.analisar_distribuicao_tamanhos(textos, classes)
        vocab_relevante = analisador.analisar_vocabulario(textos, min_freq=5)
        analisador.analisar_caracteristicas_distintas(textos, classes)

        # 5. adicionando for para validação da limpeza
        for i, (orig, limpo) in enumerate(zip(textos_originais, textos)):
            if not analisador.validar_limpeza(orig, limpo):
                logging.warning(
                    f"Possível limpeza excessiva detectada no texto {i+1}")

        # 6. adicioanndo qvetorização e divisão dos dados
        logging.info("Iniciando vetorização e divisão dos dados...")
        processador = ProcessadorVetorial(
            min_df=3,              # frequência mínima dos termos
            max_df=0.95,           # frequência máxima (%)
            ngram_range=(1, 2),    # uni e bigramas
            max_features=100000     # tamanho máximo do vocabulário
        )

        # 7. realizando a vetorização e divisão
        X_train, X_test, y_train, y_test = processador.vetorizar_e_dividir(
            textos=textos,
            classes=classes,
            test_size=0.3,         # 30% para teste
            random_state=42        # semente para reprodutibilidade
        )

        # 8. gerando visualizações da vetorização
        processador.visualizar_importancia_termos()

        # 9. salvando métricas de distribuição
        salvar_metricas_distribuicao(
            textos, classes, diretorios['graficos'])

        # 10. salvando informações sobre a divisão dos dados
        with open(os.path.join(diretorios['logs'], 'divisao_dados.log'), 'w') as f:
            f.write("=== Informações sobre a Divisão dos Dados ===\n")
            f.write(f"Total de documentos: {len(classes)}\n")
            f.write(f"Documentos no conjunto de treino: {len(y_train)}\n")
            f.write(f"Documentos no conjunto de teste: {len(y_test)}\n")
            f.write(f"\nDimensões da matriz de treino: {X_train.shape}\n")
            f.write(f"Dimensões da matriz de teste: {X_test.shape}\n")

        # 11. retornando os dados processados
        return X_train, X_test, y_train, y_test, processador.vectorizer.get_feature_names_out()

    except Exception as e:
        logging.error(f"Erro durante a execução: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, vocabulario = main()
        logging.info("Pipeline completo executado com sucesso!")
    except Exception as e:
        logging.error(f"Erro na execução do script: {str(e)}")
