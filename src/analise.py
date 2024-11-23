import logging
from collections import Counter
from typing import List, Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import os


class AnalisadorTextos:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analisar_distribuicao_tamanhos(self, textos: List[str], classes: List[str]) -> None:
        tamanhos = [len(texto.split()) for texto in textos]
        df = pd.DataFrame({'tamanho': tamanhos, 'genero': classes})

        estatisticas = df.groupby('genero')['tamanho'].describe()
        estatisticas.to_csv('estatisticas_tamanho.csv')

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='genero', y='tamanho', data=df)
        plt.title('Distribuição do Tamanho dos Textos por Gênero')
        plt.ylabel('Número de Palavras')
        plt.savefig('distribuicao_tamanhos.png')
        plt.close()

        self.logger.info("Análise de distribuição de tamanhos concluída")

    def analisar_vocabulario(self, textos: List[str], min_freq: int = 5) -> Dict:
        todas_palavras = []
        for texto in textos:
            palavras = word_tokenize(texto.lower())
            todas_palavras.extend(palavras)

        freq_palavras = Counter(todas_palavras)
        vocab_relevante = {palavra: freq for palavra,
                           freq in freq_palavras.items() if freq >= min_freq}

        with open('analise_vocabulario.txt', 'w', encoding='utf-8') as f:
            f.write(f"Tamanho total do vocabulário: {len(freq_palavras)}\n")
            f.write(
                f"Vocabulário relevante (freq >= {min_freq}): {len(vocab_relevante)}\n\n")
            f.write("Top 100 palavras mais frequentes:\n")
            for palavra, freq in sorted(vocab_relevante.items(), key=lambda x: x[1], reverse=True)[:100]:
                f.write(f"{palavra}: {freq}\n")

        return vocab_relevante

    def analisar_caracteristicas_distintas(self, textos: List[str], classes: List[str]) -> None:
        generos_unicos = set(classes)
        vocab_por_genero = {}

        for genero in generos_unicos:
            textos_genero = (texto for texto, classe in zip(
                textos, classes) if classe == genero)
            todas_palavras = []
            for texto in textos_genero:
                palavras = word_tokenize(texto.lower())
                todas_palavras.extend(palavras)

            vocab_por_genero[genero] = Counter(todas_palavras)

        with open('caracteristicas_distintas.txt', 'w', encoding='utf-8') as f:
            for genero in generos_unicos:
                f.write(f"\nPalavras características do gênero {genero}:\n")
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

                    if eh_caracteristica and freq >= 5:
                        palavras_caracteristicas.append(
                            (palavra, freq_relativa_atual))

                for palavra, freq in sorted(palavras_caracteristicas, key=lambda x: x[1], reverse=True)[:20]:
                    f.write(f"{palavra}: {freq:.4f}\n")

    def validar_limpeza(self, texto_original: str, texto_limpo: str) -> bool:
        palavras_originais = set(word_tokenize(texto_original.lower()))
        palavras_limpas = set(word_tokenize(texto_limpo.lower()))

        proporcao_mantida = len(
            palavras_limpas) / len(palavras_originais) if palavras_originais else 0
        return proporcao_mantida >= 0.4

    def gerar_wordcloud(self, vocabulario: Dict[str, int]) -> None:
        freq_palavras = Counter(vocabulario)
        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate_from_frequencies(freq_palavras)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud do Vocabulário')
        plt.show()


def salvar_metricas_distribuicao(textos, classes, dir_graficos):
    tamanhos = [len(texto.split()) for texto in textos]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=classes, y=tamanhos)
    plt.title('Distribuição do Tamanho dos Textos por Gênero')
    plt.xlabel('Gênero')
    plt.ylabel('Número de Palavras')
    plt.savefig(os.path.join(dir_graficos, 'distribuicao_tamanhos.png'))
    plt.close()


class AnalisadorTextos:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analisar_distribuicao_tamanhos(self, textos: List[str], classes: List[str]) -> None:
        tamanhos = [len(texto.split()) for texto in textos]
        df = pd.DataFrame({'tamanho': tamanhos, 'genero': classes})

        estatisticas = df.groupby('genero')['tamanho'].describe()
        estatisticas.to_csv('estatisticas_tamanho.csv')

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='genero', y='tamanho', data=df)
        plt.title('Distribuição do Tamanho dos Textos por Gênero')
        plt.ylabel('Número de Palavras')
        plt.savefig('distribuicao_tamanhos.png')
        plt.close()

        self.logger.info("Análise de distribuição de tamanhos concluída")

    def analisar_vocabulario(self, textos: List[str], min_freq: int = 5) -> Dict:
        todas_palavras = []
        for texto in textos:
            palavras = word_tokenize(texto.lower())
            todas_palavras.extend(palavras)

        freq_palavras = Counter(todas_palavras)
        vocab_relevante = {palavra: freq for palavra,
                           freq in freq_palavras.items() if freq >= min_freq}

        with open('analise_vocabulario.txt', 'w', encoding='utf-8') as f:
            f.write(f"Tamanho total do vocabulário: {len(freq_palavras)}\n")
            f.write(
                f"Vocabulário relevante (freq >= {min_freq}): {len(vocab_relevante)}\n\n")
            f.write("Top 100 palavras mais frequentes:\n")
            for palavra, freq in sorted(vocab_relevante.items(), key=lambda x: x[1], reverse=True)[:100]:
                f.write(f"{palavra}: {freq}\n")

        return vocab_relevante

    def analisar_caracteristicas_distintas(self, textos: List[str], classes: List[str]) -> None:
        generos_unicos = set(classes)
        vocab_por_genero = {}

        for genero in generos_unicos:
            textos_genero = (texto for texto, classe in zip(
                textos, classes) if classe == genero)
            todas_palavras = []
            for texto in textos_genero:
                palavras = word_tokenize(texto.lower())
                todas_palavras.extend(palavras)

            vocab_por_genero[genero] = Counter(todas_palavras)

        with open('caracteristicas_distintas.txt', 'w', encoding='utf-8') as f:
            for genero in generos_unicos:
                f.write(f"\nPalavras características do gênero {genero}:\n")
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

                    if eh_caracteristica and freq >= 5:
                        palavras_caracteristicas.append(
                            (palavra, freq_relativa_atual))

                for palavra, freq in sorted(palavras_caracteristicas, key=lambda x: x[1], reverse=True)[:20]:
                    f.write(f"{palavra}: {freq:.4f}\n")

    def validar_limpeza(self, texto_original: str, texto_limpo: str) -> bool:
        palavras_originais = set(word_tokenize(texto_original.lower()))
        palavras_limpas = set(word_tokenize(texto_limpo.lower()))

        proporcao_mantida = len(
            palavras_limpas) / len(palavras_originais) if palavras_originais else 0
        return proporcao_mantida >= 0.4

    def gerar_wordcloud(self, vocabulario: Dict[str, int]) -> None:
        freq_palavras = Counter(vocabulario)
        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate_from_frequencies(freq_palavras)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud do Vocabulário')
        plt.show()


def salvar_metricas_distribuicao(textos, classes, dir_graficos):
    tamanhos = [len(texto.split()) for texto in textos]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=classes, y=tamanhos)
    plt.title('Distribuição do Tamanho dos Textos por Gênero')
    plt.xlabel('Gênero')
    plt.ylabel('Número de Palavras')
    plt.savefig(os.path.join(dir_graficos, 'distribuicao_tamanhos.png'))
    plt.close()
