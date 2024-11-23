import os
import json
import logging
from datetime import datetime
from collections import Counter
from typing import Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from scipy import stats
import numpy as np


class OtimizadorModelos:
    def __init__(self, X_train, y_train, X_test, y_test, desired_folds=10):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_folds = self.obter_minimo_folds(y_train, desired_folds)
        self.logger = logging.getLogger(__name__)

        if self.n_folds < 2:
            logging.error(
                f"n_folds={self.n_folds} é inválido para a validação cruzada com as amostras atuais.")
            return

        # Configurando métricas de avaliação
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=1),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=1),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=1)
        }

        # Dicionário para armazenar resultados
        self.resultados = {}

        # Configurando grids de hiperparâmetros
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

    def obter_minimo_folds(self, y_train, desired_folds=10):
        contagem_classes = Counter(y_train)
        min_amostras = min(contagem_classes.values())
        return min(desired_folds, min_amostras)

    def otimizar_modelo(self, nome_modelo: str, modelo, param_grid: Dict):
        logging.info(f"Iniciando otimização do modelo: {nome_modelo}")

        try:
            grid_search = GridSearchCV(
                estimator=modelo,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.n_folds,
                n_jobs=-1,
                refit='f1_macro',
                return_train_score=True,
                verbose=1
            )

            grid_search.fit(self.X_train, self.y_train)

            self.resultados[nome_modelo] = {
                'best_estimator': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }

            logging.info(f"Otimização concluída para {nome_modelo}")

        except Exception as e:
            logging.error(
                f"Erro na otimização do modelo {nome_modelo}: {str(e)}")

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
            self.otimizar_modelo(nome, modelo, self.param_grids.get(nome, {}))

    def avaliar_significancia(self):
        """Realiza o teste de Friedman com número igual de amostras."""
        try:
            metricas = ['accuracy', 'precision_macro',
                        'recall_macro', 'f1_macro']
            dados = {metrica: [] for metrica in metricas}
            modelos = list(self.resultados.keys())

            # Encontrar o menor número de amostras entre todos os modelos
            min_samples = float('inf')
            for modelo in modelos:
                for metrica in metricas:
                    scores = self.resultados[modelo]['cv_results'][f'mean_test_{metrica}']
                    min_samples = min(min_samples, len(scores))

            # Coletar scores com mesmo tamanho
            scores_padronizados = []
            for modelo in modelos:
                modelo_scores = []
                for metrica in metricas:
                    scores = self.resultados[modelo]['cv_results'][f'mean_test_{metrica}']
                    modelo_scores.extend(scores[:min_samples])
                scores_padronizados.append(modelo_scores)

            if len(scores_padronizados) >= 2:
                stat, p = stats.friedmanchisquare(*scores_padronizados)

                resultados = {
                    'statistic': float(stat),
                    'p_value': float(p),
                    'n_samples': min_samples
                }

                self.logger.info(
                    f"Teste de Friedman: estatística={stat:.4f}, p-valor={p:.4f}, "
                    f"n_amostras={min_samples}"
                )
                return resultados
            else:
                self.logger.warning(
                    "Número insuficiente de modelos para teste de Friedman")
                return None

        except Exception as e:
            self.logger.error(
                f"Erro durante o teste de significância: {str(e)}")
            return None

    def gerar_graficos_comparativos(self, diretorio_saida: str = 'resultados'):
        os.makedirs(diretorio_saida, exist_ok=True)

        metricas_df = []
        for modelo, resultado in self.resultados.items():
            # Usar cv_results em vez de scores_cv
            for metrica in ['mean_test_accuracy', 'mean_test_precision_macro',
                            'mean_test_recall_macro', 'mean_test_f1_macro']:
                if metrica in resultado['cv_results']:
                    metricas_df.append({
                        'modelo': modelo,
                        'metrica': metrica.replace('mean_test_', ''),
                        'valor': resultado['cv_results'][metrica].mean(),
                        'std': resultado['cv_results'][metrica].std()
                    })

        df = pd.DataFrame(metricas_df)
        logging.info(
            f"Colunas do DataFrame para plotagem: {df.columns.tolist()}")

        if 'modelo' not in df.columns:
            logging.error("A coluna 'modelo' não existe no DataFrame.")
            return

        # Plotagem
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x='modelo',
            y='valor',
            hue='metrica',
            capsize=0.1
        )
        plt.title('Comparação de Métricas entre Modelos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(diretorio_saida, 'comparacao_modelos.png'))
        plt.close()

    def salvar_resultados(self, diretorio_saida: str = 'resultados'):
        """
        Salva os resultados da otimização de forma segura.
        """
        os.makedirs(diretorio_saida, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Preparando resultados para serialização
        resultados_serializaveis = {}
        for modelo, resultado in self.resultados.items():
            if not resultado:  # Pula se resultado for None
                continue

            try:
                resultados_serializaveis[modelo] = {
                    'melhores_params': resultado.get('best_params', {}),
                    'melhor_score': float(resultado.get('best_score', 0.0)),
                    'metricas': {}
                }

                # Extrair métricas dos resultados CV de forma segura
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

        # Salvando em JSON
        try:
            caminho_json = os.path.join(
                diretorio_saida, f'resultados_otimizacao_{timestamp}.json')
            with open(caminho_json, 'w') as f:
                json.dump(resultados_serializaveis, f, indent=4)
        except Exception as e:
            logging.error(f"Erro ao salvar JSON: {str(e)}")

        # Convertendo para DataFrame e salvando em CSV
        try:
            rows = []
            for modelo, resultado in resultados_serializaveis.items():
                row = {'modelo': modelo}
                # Adiciona parâmetros
                row.update(
                    {f'param_{k}': v for k, v in resultado['melhores_params'].items()})
                # Adiciona métricas
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
