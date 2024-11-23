import logging
from collections import Counter
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


class ProcessadorVetorial:
    def __init__(self,
                 min_df: int = 3,
                 max_df: float = 0.95,
                 ngram_range: Tuple = (1, 2),
                 max_features: int = 50000,
                 balanceamento: str = 'combinado'):
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words='english'
        )
        self.balanceamento = balanceamento
        self.logger = logging.getLogger(__name__)

    def _aplicar_balanceamento(self, X, y, random_state):
        """
        Aplica balanceamento adaptativo para garantir distribuição igual entre classes.
        """
        try:
            contagem_classes = Counter(y)

            #  Usando ceil em vez de divisão inteira
            tamanho_alvo = ceil(
                sum(contagem_classes.values()) / len(contagem_classes))

            # Garantindo que o tamanho alvo não seja menor que a maior classe
            tamanho_alvo = max(tamanho_alvo, max(contagem_classes.values()))

            strategy = {
                classe: tamanho_alvo for classe in contagem_classes.keys()}

            # Aplicando apenas SMOTE primeiro
            smote = SMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=min(5, min(contagem_classes.values())-1)
            )

            X_bal, y_bal = smote.fit_resample(X, y)

            # Só aplica under sampling se necessário
            if len(y_bal) > tamanho_alvo * len(contagem_classes):
                under = RandomUnderSampler(
                    sampling_strategy=strategy,
                    random_state=random_state
                )
                X_bal, y_bal = under.fit_resample(X_bal, y_bal)

            self.logger.info(f"Contagem original: {contagem_classes}")
            self.logger.info(f"Tamanho alvo: {tamanho_alvo}")
            self.logger.info(f"Estratégia: {strategy}")

            return X_bal, y_bal

        except Exception as e:
            self.logger.error(f"Erro no balanceamento adaptativo: {str(e)}")
            return X, y  # Retorna dados originais em caso de erro

    def vetorizar_e_dividir(self, textos: List[str], classes: List[str],
                            test_size: float = 0.3, random_state: int = 42) -> Tuple:
        try:
            # Vetorização TF-IDF - Bag of Words expandido  (n-gramas)
            X = self.vectorizer.fit_transform(textos)

            # Primeiro balanceamento antes da divisão
            X_bal, y_bal = self._aplicar_balanceamento(
                X, classes, random_state)

            # Divisão estratificada
            X_train, X_test, y_train, y_test = train_test_split(
                X_bal, y_bal,
                test_size=test_size,
                random_state=random_state,
                stratify=y_bal  # Usa dados já balanceados
            )

            # Segundo balanceamento apenas no conjunto de treino
            X_train, y_train = self._aplicar_balanceamento(
                X_train, y_train, random_state)

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

        # Comparando proporções
        for classe in set(y_train):
            diff = abs(prop_train.get(classe, 0) - prop_test.get(classe, 0))
            if diff > 5:  # Diferença maior que 5%
                self.logger.warning(
                    f"Desbalanceamento detectado na classe {classe}: "
                    f"Treino={prop_train.get(classe, 0):.1f}%, "
                    f"Teste={prop_test.get(classe, 0):.1f}%"
                )

    def _analisar_esparsidade(self, X) -> None:
        """Analisa a esparsidade da matriz TF-IDF."""
        densidade = (X.nnz / (X.shape[0] * X.shape[1])) * 100
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

        # Analisando n-gramas
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

            # Ordenando por importância
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


def verificar_distribuicao(y):
    contagem = Counter(y)
    logging.info(f"Distribuição das classes: {contagem}")
