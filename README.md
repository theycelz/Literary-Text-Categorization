# Classificação de Textos Literários

## Objetivo
Este projeto tem como objetivo explorar a classificação de textos de diferentes estilos literários, como poesia, prosa e jornalismo. O desafio envolve a extração de textos de arquivos PDF, pré-processamento, e transformação dos textos em vetores utilizando o modelo **Bag of Words (BoW)**. O projeto precisa garantir que todos os documentos estejam na mesma língua para evitar inconsistências na classificação.

## Instruções
- **Seleção dos Estilos Literários**: O grupo deve selecionar **três estilos literários** (ex.: poesia, prosa e jornalismo).
- **Extração de Texto**: Todos os textos devem ser extraídos de arquivos **PDF** e convertidos para **TXT** antes do processamento.
- **Limpeza de Dados**: A remoção de **stopwords** é obrigatória antes de gerar os vetores BoW.
- **Consistência de Dados**: Cada estilo literário deve conter pelo menos **100 exemplos**.
- **Linguagem Única**: Todos os textos devem estar na mesma língua.

## Requisitos
- **Fontes de Dados**:
  - **Poesia**: Project Gutenberg.
  - **Prosa**: Obras literárias no Project Gutenberg.
  - **Jornalismo**: Artigos do Corpus of Contemporary American English (COCA) ou notícias de sites confiáveis.

## Metodologia
O procedimento envolve a aplicação de **algoritmos de aprendizado supervisionado** para classificar textos, como Árvores de Decisão, K-Nearest Neighbors, Naïve Bayes, Regressão Logística e Redes Neurais. A avaliação será feita usando **acurácia** e **F1-score**, utilizando validação cruzada estratificada com 10 folds.


