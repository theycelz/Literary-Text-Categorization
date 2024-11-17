from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, List
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ProcessadorVetorial:
    def __init__(self,
                 min_df: int = 3,
                 max_df: float = 0.95,
                 ngram_range: Tuple = (1, 2),
                 max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english'
        )
        self.logger = logging.getLogger(__name__)

    def vetorizar_e_dividir(self,
                            textos: List[str],
                            classes: List[str],
                            test_size: float = 0.3,
                            random_state: int = 42) -> Tuple:
        """
        Vetoriza os textos usando TF-IDF e divide em conjuntos de treino e teste.
        """
        try:
            # vetorização TF-IDF
            X = self.vectorizer.fit_transform(textos)

            # gerando relatório do vocabulário
            self._analisar_vocabulario()

            # divisão estratificada
            X_train, X_test, y_train, y_test = train_test_split(
                X, classes,
                test_size=test_size,
                random_state=random_state,
                stratify=classes
            )

            # validando o balanceamento
            self._validar_balanceamento(y_train, y_test)

            # analisando esparsidade
            self._analisar_esparsidade(X_train)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Erro na vetorização/divisão: {str(e)}")
            raise

    def _validar_balanceamento(self, y_train: List[str], y_test: List[str]) -> None:
        """Valida o balanceamento das classes nos conjuntos."""
        def calcular_proporcoes(y):
            total = len(y)
            return {classe: (count/total)*100
                    for classe, count in Counter(y).items()}

        prop_train = calcular_proporcoes(y_train)
        prop_test = calcular_proporcoes(y_test)

        # comparando proporções
        for classe in set(y_train):
            diff = abs(prop_train[classe] - prop_test[classe])
            if diff > 5:  # diferença maior que 5%
                self.logger.warning(
                    f"Desbalanceamento detectado na classe {classe}: "
                    f"Treino={prop_train[classe]:.1f}%, "
                    f"Teste={prop_test[classe]:.1f}%"
                )

    def _analisar_esparsidade(self, X) -> None:
        """Analisa a esparsidade da matriz TF-IDF."""
        densidade = (X.nnz / np.prod(X.shape)) * 100
        elementos_nao_zero = X.nnz

        self.logger.info(
            f"Densidade da matriz: {densidade:.2f}%\n"
            f"Elementos não-zero: {elementos_nao_zero}"
        )

        if densidade < 0.1:
            self.logger.warning(
                "Matriz muito esparsa. Considere reduzir max_features "
                "ou aumentar min_df"
            )

    def _analisar_vocabulario(self) -> None:
        """Analisa e registra estatísticas do vocabulário."""
        vocabulario = self.vectorizer.get_feature_names_out()

        # analisando n-gramas
        unigramas = sum(1 for termo in vocabulario if len(termo.split()) == 1)
        bigramas = sum(1 for termo in vocabulario if len(termo.split()) == 2)

        self.logger.info(
            f"Tamanho do vocabulário: {len(vocabulario)}\n"
            f"Unigramas: {unigramas}\n"
            f"Bigramas: {bigramas}"
        )

    def visualizar_importancia_termos(self, n_termos: int = 20) -> None:
        """Visualiza os termos mais importantes do vocabulário."""
        try:
            termos = self.vectorizer.get_feature_names_out()

            # ordenando por importância
            indices = np.argsort(self.vectorizer.idf_)[-n_termos:]

            plt.figure(figsize=(12, 6))
            sns.barplot(
                x=[termos[i] for i in indices],
                y=[self.vectorizer.idf_[i] for i in indices]
            )
            plt.xticks(rotation=45, ha='right')
            plt.title('Termos Mais Importantes do Vocabulário')
            plt.tight_layout()
            plt.savefig('termos_importantes.png')
            plt.close()

        except Exception as e:
            self.logger.error(f"Erro na visualização: {str(e)}")
