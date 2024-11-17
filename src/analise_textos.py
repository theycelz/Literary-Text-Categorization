import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
import seaborn as sns
import logging
from typing import List, Dict, Tuple
import numpy as np


class AnalisadorTextos:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analisar_distribuicao_tamanhos(self, textos: List[str], classes: List[str]) -> None:
        """Analisa e plota a distribuição do tamanho dos textos por gênero."""
        tamanhos = [len(texto.split()) for texto in textos]
        df = pd.DataFrame({
            'tamanho': tamanhos,
            'genero': classes
        })

        # salvando algumas estatísticas básicas
        estatisticas = df.groupby('genero')['tamanho'].describe()
        estatisticas.to_csv('estatisticas_tamanho.csv')

        # plotando distribuição
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='genero', y='tamanho', data=df)
        plt.title('Distribuição do Tamanho dos Textos por Gênero')
        plt.ylabel('Número de Palavras')
        plt.savefig('distribuicao_tamanhos.png')
        plt.close()

        self.logger.info("Análise de distribuição de tamanhos concluída")

    def analisar_vocabulario(self, textos: List[str], min_freq: int = 5) -> Dict:
        """Analisa o vocabulário e frequência dos termos."""
        todas_palavras = []
        for texto in textos:
            palavras = word_tokenize(texto.lower())
            todas_palavras.extend(palavras)

        freq_palavras = Counter(todas_palavras)

        # filtrando palavras com frequência mínima
        vocab_relevante = {palavra: freq for palavra, freq in freq_palavras.items()
                           if freq >= min_freq}

        # aqui estamos salvando análise do vocabulário
        with open('analise_vocabulario.txt', 'w', encoding='utf-8') as f:
            f.write(f"Tamanho total do vocabulário: {len(freq_palavras)}\n")
            f.write(f"Vocabulário relevante (freq >= {min_freq}): {
                    len(vocab_relevante)}\n\n")
            f.write("Top 100 palavras mais frequentes:\n")
            for palavra, freq in sorted(vocab_relevante.items(),
                                        key=lambda x: x[1], reverse=True)[:100]:
                f.write(f"{palavra}: {freq}\n")

        return vocab_relevante

    def analisar_caracteristicas_distintas(self, textos: List[str],
                                           classes: List[str]) -> None:
        """Analisa características distintivas por gênero."""
        generos_unicos = set(classes)
        vocab_por_genero = {}

        # calculando vocabulário específico por gênero
        for genero in generos_unicos:
            textos_genero = [texto for texto, classe in zip(textos, classes)
                             if classe == genero]
            todas_palavras = []
            for texto in textos_genero:
                palavras = word_tokenize(texto.lower())
                todas_palavras.extend(palavras)

            vocab_por_genero[genero] = Counter(todas_palavras)

        # TODO: identificar palavras características por gênero
        with open('caracteristicas_distintas.txt', 'w', encoding='utf-8') as f:
            for genero in generos_unicos:
                f.write(f"\nPalavras características do gênero {genero}:\n")
                # comparando frequências relativas entre gêneros
                palavras_caracteristicas = []
                vocab_atual = vocab_por_genero[genero]

                for palavra, freq in vocab_atual.items():
                    freq_relativa_atual = freq / sum(vocab_atual.values())
                    eh_caracteristica = True

                    for outro_genero in generos_unicos:
                        if outro_genero != genero:
                            outro_vocab = vocab_por_genero[outro_genero]
                            freq_outro = outro_vocab.get(palavra, 0)
                            freq_relativa_outro = freq_outro / \
                                sum(outro_vocab.values())

                            if freq_relativa_outro >= freq_relativa_atual:
                                eh_caracteristica = False
                                break

                    if eh_caracteristica and freq >= 5:  # filtrando palavras raras
                        palavras_caracteristicas.append(
                            (palavra, freq_relativa_atual))

                # escrevendo top palavras características
                for palavra, freq in sorted(palavras_caracteristicas,
                                            key=lambda x: x[1], reverse=True)[:20]:
                    f.write(f"{palavra}: {freq:.4f}\n")
