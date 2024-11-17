import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import json
from datetime import datetime
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import logging
from scipy import stats


class OtimizadorModelos:
    def __init__(self, X_train, y_train, X_test, y_test, n_folds=10):
        """
        Inicializa o otimizador com dados de treino e teste.

        Args:
            X_train: features de treino
            y_train: labels de treino
            X_test: features de teste
            y_test: lbels de teste
            n_folds: número de folds para validação cruzada
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_folds = n_folds

        # aqui estamos configurando métricas de avaliação
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro'),
            'recall_macro': make_scorer(recall_score, average='macro'),
            'f1_macro': make_scorer(f1_score, average='macro')
        }

        # definindo dicionário para armazenar resultados
        self.resultados = {}

        # configurando grids de hiperparâmetros
        self.param_grids = {
            'DecisionTree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'cosine']
            },
            'NaiveBayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'class_weight': [None, 'balanced'],
                'solver': ['lbfgs', 'newton-cg'],
                'max_iter': [1000]
            },
            'MLP': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'early_stopping': [True]
            }
        }

        def otimizar_modelo(self, nome_modelo: str, modelo, param_grid: Dict):
            """
            Otimiza um modelo usando GridSearchCV.

            Args:
                nome_modelo: Nome do modelo
                modelo: Instância do modelo
                param_grid: Grid de hiperparâmetros
            """
            logging.info(f"Iniciando otimização do modelo: {nome_modelo}")

            try:
                # configurando GridSsearchCV
                grid_search = GridSearchCV(
                    estimator=modelo,
                    param_grid=param_grid,
                    scoring=self.scoring,
                    cv=self.n_folds,
                    n_jobs=-1,
                    refit='f1_macro',  # Usando F1 macro como métrica principal
                    return_train_score=True,
                    verbose=1
                )

               # realizando busca
                grid_search.fit(self.X_train, self.y_train)

                # salvando os resultados
                self.resultados[nome_modelo] = {
                    'melhores_params': grid_search.best_params_,
                    'melhor_score': grid_search.best_score_,
                    'scores_cv': {
                        metric: {
                            'mean': grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_],
                            'std': grid_search.cv_results_[f'std_test_{metric}'][grid_search.best_index_]
                        }
                        for metric in self.scoring.keys()
                    },
                    'modelo_otimizado': grid_search.best_estimator_
                }

                logging.info(f"Otimização concluída para {nome_modelo}")

            except Exception as e:
                logging.error(f"Erro na otimização do modelo {
                    nome_modelo}: {str(e)}")

        def otimizar_todos_modelos(self):
            """Otimiza todos os modelos definidos."""
            modelos = {
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'NaiveBayes': MultinomialNB(),
                'LogisticRegression': LogisticRegression(random_state=42),
                'MLP': MLPClassifier(random_state=42, max_iter=1000)
            }
            for nome, modelo in modelos.items():
                self.otimizar_modelo(nome, modelo, self.param_grids[nome])

        def avaliar_significancia(self):
            """Realiza teste estatístico para comparar modelos."""
            scores = {}
            for nome, resultado in self.resultados.items():
                modelo = resultado['modelo_otimizado']
                pred = modelo.predict(self.X_test)
                scores[nome] = f1_score(self.y_test, pred, average='macro')

            # Realizando teste de Friedman
            nomes_modelos = list(scores.keys())
            valores_f1 = list(scores.values())

            _, p_value = stats.friedmanchisquare(*valores_f1)

            return {
                'scores': scores,
                'p_value': p_value
            }
