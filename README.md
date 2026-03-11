# Classificação de Textos Literários

Classificação automatizada de textos literários em gêneros (Horror, Poesia, Romance) utilizando aprendizado de máquina supervisionado em um corpus de 300+ obras de domínio público do Project Gutenberg.

## Visão Geral

Pipeline completo de NLP que:

1. **Extrai** texto de arquivos PDF com `pdfminer`
2. **Pré-processa** com tokenização, remoção de stopwords e validação de idioma
3. **Vetoriza** usando TF-IDF com uni/bigramas e balanceamento de classes (SMOTE)
4. **Classifica** com 5 modelos: Árvore de Decisão, KNN, Naive Bayes, Regressão Logística e MLP
5. **Otimiza** hiperparâmetros via GridSearchCV com validação cruzada estratificada
6. **Avalia** com acurácia, precisão, recall, F1-score (macro) e teste de significância de Friedman

## Estrutura do Projeto

```
src/
  main.py            # Ponto de entrada do pipeline
  processamento.py   # Extração de PDFs, limpeza de texto, detecção de idioma
  vetorizacao.py      # Vetorização TF-IDF, balanceamento com SMOTE
  analise.py          # Análise de textos, estatísticas de vocabulário, wordcloud
  classificador.py    # Treinamento e validação cruzada dos modelos
  otimizador.py       # Otimização de hiperparâmetros com GridSearchCV
  pdfsHorror/         # Corpus de horror (100 PDFs)
  pdfsPoetry/         # Corpus de poesia (100 PDFs)
  pdfsRomance/        # Corpus de romance (100 PDFs)
```

## Como Executar

```bash
pip install -r requirements.txt
```

```bash
cd src && python main.py
```

## Dataset

Todos os textos estão em inglês, obtidos do [Project Gutenberg](https://www.gutenberg.org/). Cada gênero contém ~100 obras, incluindo:

| Gênero  | Exemplos                                               |
|---------|--------------------------------------------------------|
| Horror  | Dracula, Frankenstein, The Turn of the Screw            |
| Poesia  | Leaves of Grass, Paradise Lost, The Waste Land          |
| Romance | Don Quixote, Ivanhoe, Romeo and Juliet                  |

## Modelos

| Modelo               | Abordagem                                            |
|----------------------|------------------------------------------------------|
| Árvore de Decisão     | Pesos balanceados, ajuste de profundidade/folhas     |
| KNN                   | Distância cosseno, otimização de k                   |
| Naive Bayes           | Multinomial com suavização alpha                     |
| Regressão Logística   | Regularização L2, pesos balanceados                  |
| MLP                   | Early stopping, taxa de aprendizado adaptativa       |

Todos os modelos são avaliados com validação cruzada estratificada de 10 folds e comparados pelo teste estatístico de Friedman.

## Licença

MIT
