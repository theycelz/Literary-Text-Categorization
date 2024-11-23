import os
import json
import logging
from datetime import datetime
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
import numpy as np
import pandas as pd


class ClassificadorGeneros:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.scoring = {
            'accuracy': 'accuracy',
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=1),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=1),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=1)
        }

        self.resultados = {}

    def configurar_classificadores(self):
        self.classificadores = {
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='cosine',
                n_jobs=-1
            ),
            'NaiveBayes': MultinomialNB(
                alpha=1.0
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
        n_folds = min(5, min(Counter(self.y_train).values()))

        if n_folds < 2:
            logging.error("Amostras insuficientes para validação cruzada")
            return

        for nome, clf in self.classificadores.items():
            logging.info(f"Iniciando treinamento do classificador: {nome}")

            try:
                cv_results = cross_validate(
                    clf,
                    self.X_train,
                    self.y_train,
                    cv=n_folds,
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
                logging.error(
                    f"Erro no treinamento do classificador {nome}: {str(e)}")
                continue

    def salvar_resultados(self, diretorio_saida: str = 'resultados'):
        os.makedirs(diretorio_saida, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        resultados_serializaveis = {}
        for modelo, resultado in self.resultados.items():
            if not resultado:
                continue

            try:
                resultados_serializaveis[modelo] = {
                    'melhores_params': resultado.get('best_params', {}),
                    'melhor_score': float(resultado.get('best_score', 0.0)),
                    'metricas': {}
                }

                cv_results = resultado.get('cv_results', {})
                for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
                    metric_key = f'mean_test_{metric}'
                    if metric_key in cv_results:
                        resultados_serializaveis[modelo]['metricas'][metric] = {
                            'mean': float(np.mean(cv_results[metric_key])),
                            'std': float(np.std(cv_results[metric_key]))
                        }

            except Exception as e:
                logging.warning(
                    f"Erro ao processar resultados do modelo {modelo}: {str(e)}")
                continue

        try:
            caminho_json = os.path.join(
                diretorio_saida, f'resultados_otimizacao_{timestamp}.json')
            with open(caminho_json, 'w') as f:
                json.dump(resultados_serializaveis, f, indent=4)
        except Exception as e:
            logging.error(f"Erro ao salvar JSON: {str(e)}")

        try:
            rows = []
            for modelo, resultado in resultados_serializaveis.items():
                row = {'modelo': modelo}
                row.update(
                    {f'param_{k}': v for k, v in resultado['melhores_params'].items()})
                for metrica, valores in resultado['metricas'].items():
                    row[f'{metrica}_mean'] = valores['mean']
                    row[f'{metrica}_std'] = valores['std']
                rows.append(row)

            if rows:
                df_resultados = pd.DataFrame(rows)
                caminho_csv = os.path.join(
                    diretorio_saida, f'resultados_otimizacao_{timestamp}.csv')
                df_resultados.to_csv(caminho_csv, index=False)

        except Exception as e:
            logging.error(f"Erro ao salvar CSV: {str(e)}")

        logging.info(f"Resultados salvos em {diretorio_saida}")
