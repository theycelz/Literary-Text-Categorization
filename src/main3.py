import os
import re
import json

import logging
import multiprocessing
import cProfile
import pstats
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from scipy import stats
import numpy as np
import chardet
from math import ceil
from multiprocessing import Lock, Manager
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


logging.basicConfig(
    filename='processamento.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

STOP_WORDS = set(stopwords.words('english'))


def processar_pdf(args):
    arquivo, classe, preservar_palavras, diretorio_raiz = args
    try:
        logging.info("Iniciando processamento")
        texto_extraido = pdf_para_txt(arquivo)
        valido, motivo = validar_texto(texto_extraido)
        if not valido:
            logging.warning(f"Texto inválido em {arquivo}: {motivo}", extra={
                            'nome_funcao': 'processar_pdf'})
            return None
        texto_limpo, texto_original = limpar_texto(
            texto_extraido, preservar_palavras)
        if (texto_limpo):
            nome_arquivo_txt = os.path.splitext(os.path.basename(arquivo))[0]
            if salvar_texto_em_arquivo(nome_arquivo_txt, texto_limpo, texto_original, diretorio_raiz, classe):
                return (texto_limpo, texto_original, classe)
    except Exception as e:
        logging.error(f"Erro: {arquivo} - {str(e)}")
    return None


def processar_pdfs(diretorio_raiz: str, diretorios_pdfs: Dict[str, str]):
    if not os.path.isdir(diretorio_raiz):
        raise ValueError(f"Diretório raiz inválido: {diretorio_raiz}")

    for classe, caminho in diretorios_pdfs.items():
        if not os.path.isdir(caminho):
            raise ValueError(
                f"Diretório inválido para classe {classe}: {caminho}")

    if not diretorios_pdfs:
        raise ValueError("Nenhum diretório de PDFs fornecido")
    textos_limpos = []
    textos_originais = []
    classes = []
    manager = Manager()
    estatisticas = manager.dict({
        'total_processado': 0,
        'sucessos': 0,
        'falhas': 0,
        'por_classe': manager.dict({
            classe: manager.dict({'processados': 0, 'falhas': 0})
            for classe in diretorios_pdfs.keys()
        })
    })

    lock = Lock()

    def atualizar_estatisticas(resultado, classe):
        with lock:
            if resultado:
                estatisticas['sucessos'] += 1
                estatisticas['por_classe'][classe]['processados'] += 1
            else:
                estatisticas['falhas'] += 1
                estatisticas['por_classe'][classe]['falhas'] += 1

    # Palavras importantes a serem preservadas por gênero
    palavras_preservar_dict = {
        'horror': {'fear', 'dark', 'blood', 'death', 'night', 'ghost', 'shadow'},
        'poetry': {'love', 'heart', 'soul', 'dream', 'light', 'sky', 'wind'},
        'romance': {'love', 'heart', 'kiss', 'smile', 'eyes', 'touch', 'feel'}
    }

    # Coletando todas os PDFs para processamento
    tarefas = []
    for classe, caminho_pdfs in diretorios_pdfs.items():
        arquivos_pdf = [os.path.join(caminho_pdfs, f) for f in os.listdir(
            caminho_pdfs) if f.endswith('.pdf')]
        for arquivo_pdf in arquivos_pdf:
            tarefas.append((arquivo_pdf, classe, palavras_preservar_dict.get(
                classe, set()), diretorio_raiz))

    # Processando PDFs em paralelo
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        resultados = pool.map(processar_pdf, tarefas)

    # Coletando resultados válidos
    for resultado in resultados:
        if resultado:
            texto_limpo, texto_original, classe = resultado
            textos_limpos.append(texto_limpo)
            textos_originais.append(texto_original)
            classes.append(classe)
            estatisticas['sucessos'] += 1
            estatisticas['por_classe'][classe]['processados'] += 1
        else:
            estatisticas['falhas'] += 1

    # Log final das estatísticas
    logging.info("=== Estatísticas de Processamento ===")
    logging.info(f"Total processado: {len(tarefas)}")
    logging.info(f"Sucessos: {estatisticas['sucessos']}")
    logging.info(f"Falhas: {estatisticas['falhas']}")
    for classe, stats in estatisticas['por_classe'].items():
        logging.info(f"Classe {classe}: Processados {stats['processados']}, "
                     f"Falhas {stats['falhas']}")

    # Salvando métricas gerais
    metricas_gerais = {
        'total_documentos': len(tarefas),
        'documentos_processados': estatisticas['sucessos'],
        'documentos_falhos': estatisticas['falhas'],
        'taxa_sucesso': estatisticas['sucessos'] / len(tarefas) if len(tarefas) > 0 else 0
    }

    df_metricas = pd.DataFrame([metricas_gerais])
    df_metricas.to_csv(os.path.join(
        diretorio_raiz, 'metricas_gerais.csv'), index=False)

    if not textos_limpos:
        raise ValueError("Nenhum texto foi extraído com sucesso.")

    return textos_limpos, textos_originais, classes


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

        # Salvando estatísticas básicas
        estatisticas = df.groupby('genero')['tamanho'].describe()
        estatisticas.to_csv('estatisticas_tamanho.csv')

        # Plotando distribuição
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

        # Filtrando palavras com frequência mínima
        vocab_relevante = {palavra: freq for palavra, freq in freq_palavras.items()
                           if freq >= min_freq}

        # Salvando análise do vocabulário
        with open('analise_vocabulario.txt', 'w', encoding='utf-8') as f:
            f.write(f"Tamanho total do vocabulário: {len(freq_palavras)}\n")
            f.write(
                f"Vocabulário relevante (freq >= {min_freq}): {len(vocab_relevante)}\n\n")
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

        # Calculando vocabulário específico por gênero
        for genero in generos_unicos:
            textos_genero = (texto for texto, classe in zip(textos, classes)
                             if classe == genero)
            todas_palavras = []
            for texto in textos_genero:
                palavras = word_tokenize(texto.lower())
                todas_palavras.extend(palavras)

            vocab_por_genero[genero] = Counter(todas_palavras)

        # Identificando palavras características por gênero
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

                # Escrevendo top palavras características
                for palavra, freq in sorted(palavras_caracteristicas,
                                            key=lambda x: x[1], reverse=True)[:20]:
                    f.write(f"{palavra}: {freq:.4f}\n")

    def validar_limpeza(self, texto_original: str, texto_limpo: str) -> bool:
        """Valida se a limpeza não foi excessiva."""
        palavras_originais = set(word_tokenize(texto_original.lower()))
        palavras_limpas = set(word_tokenize(texto_limpo.lower()))

        # Calculando proporção de palavras mantidas
        proporcao_mantida = len(
            palavras_limpas) / len(palavras_originais) if palavras_originais else 0

        # Aferindo que se menos de 40% das palavras forem mantidas, significa limpeza excessiva
        return proporcao_mantida >= 0.4

    def gerar_wordcloud(self, vocabulario: Dict[str, int]) -> None:
        """Gera e plota uma Word Cloud do vocabulário."""
        # Frequência das palavras no vocabulário
        freq_palavras = Counter(vocabulario)

        # Gerar Word Cloud
        wordcloud = WordCloud(
            width=800, height=400, background_color='white').generate_from_frequencies(freq_palavras)

        plt.figure(figsize=(15, 7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud do Vocabulário')
        plt.show()


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
                class_weight='balanced'  # Adicionado
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # Adicionado
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='cosine',
                n_jobs=-1  # Adicionado para paralelismo
            ),
            'NaiveBayes': MultinomialNB(
                alpha=1.0
                # MultinomialNB não suporta class_weight
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100,),
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
                # MLPClassifier não suporta class_weight diretamente
            )
        }

    def treinar_e_avaliar(self):
        """
        Treina e avalia cada classificador usando validação cruzada.
        """
        # Determinar número adequado de folds
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
                    cv=n_folds,  # Usando número adaptativo de folds
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

            if rows:  # Só cria DataFrame se houver dados
                df_resultados = pd.DataFrame(rows)
                caminho_csv = os.path.join(
                    diretorio_saida, f'resultados_otimizacao_{timestamp}.csv')
                df_resultados.to_csv(caminho_csv, index=False)

        except Exception as e:
            logging.error(f"Erro ao salvar CSV: {str(e)}")

        logging.info(f"Resultados salvos em {diretorio_saida}")


class OtimizadorModelos:
    def __init__(self, X_train, y_train, X_test, y_test, desired_folds=10):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_folds = obter_minimo_folds(y_train, desired_folds)

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

    def otimizar_modelo(self, nome_modelo: str, modelo, param_grid: Dict):
        logging.info(f"Iniciando otimização do modelo: {nome_modelo}")

        try:
            grid_search = GridSearchCV(
                estimator=modelo,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.n_folds,  # Usa o número ajustado de folds
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
        """Realiza o teste de Friedman para avaliar a significância estatística entre os modelos."""
        try:
            metricas = ['accuracy', 'precision_macro',
                        'recall_macro', 'f1_macro']
            dados = {metrica: [] for metrica in metricas}
            modelos = list(self.resultados.keys())

            for metrica in metricas:
                for modelo in modelos:
                    dados[metrica].append(
                        self.resultados[modelo]['cv_results'][f'mean_test_{metrica}'])

            # Verificar se há pelo menos 3 modelos
            if len(modelos) < 3:
                logging.error(
                    "Erro durante a execução: At least 3 models must be compared for Friedman test, got fewer.")
                return None

            # Exemplo para a métrica F1
            f1_scores = dados['f1_macro']

            # Verificar se há pelo menos 3 amostras
            if len(f1_scores) < 3:
                logging.error(
                    "Erro durante a execução: At least 3 sets of samples must be given for Friedman test, got fewer.")
                return None

            # Realizar o teste de Friedman
            stat, p = stats.friedmanchisquare(
                *[dados['f1_macro'][i] for i in range(len(modelos))])

            resultados = {
                'statistic': stat,
                'p_value': p
            }

            logging.info(
                f"Resultado do teste de Friedman: Estatística={stat}, p-valor={p}")
            return resultados

        except Exception as e:
            logging.error(f"Erro durante o teste de significância: {str(e)}")
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

        # Verifique se a coluna 'modelo' existe
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

            if rows:  # Só cria DataFrame se houver dados
                df_resultados = pd.DataFrame(rows)
                caminho_csv = os.path.join(
                    diretorio_saida, f'resultados_otimizacao_{timestamp}.csv')
                df_resultados.to_csv(caminho_csv, index=False)

        except Exception as e:
            logging.error(f"Erro ao salvar CSV: {str(e)}")

        logging.info(f"Resultados salvos em {diretorio_saida}")


def criar_diretorios_saida(diretorio_raiz):
    """Cria diretórios necessários para salvar as análises."""
    diretorios = ['analises', 'graficos',
                  'logs', 'resultados', 'textos_extraidos']
    for dir_nome in diretorios:
        caminho = os.path.join(diretorio_raiz, dir_nome)
        os.makedirs(caminho, exist_ok=True)
    return {nome: os.path.join(diretorio_raiz, nome) for nome in diretorios}


def salvar_metricas_distribuicao(textos, classes, dir_graficos):
    """Salva gráficos e métricas sobre a distribuição dos textos."""
    # Calculando a distribuição do tamanho dos textos por classe
    tamanhos = [len(texto.split()) for texto in textos]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=classes, y=tamanhos)
    plt.title('Distribuição do Tamanho dos Textos por Gênero')
    plt.xlabel('Gênero')
    plt.ylabel('Número de Palavras')
    plt.savefig(os.path.join(dir_graficos, 'distribuicao_tamanhos.png'))
    plt.close()


def detectar_encoding(texto_bytes):
    """Detecta o encoding do texto."""
    resultado = chardet.detect(texto_bytes)
    return resultado['encoding']


def verificar_lingua(texto, lingua_esperada='en'):
    """Verifica se o texto está na língua esperada."""
    try:
        return detect(texto) == lingua_esperada
    except Exception as e:
        logging.error(f"Erro ao detectar língua: {str(e)}")
        return False


def pdf_para_txt(caminho_pdf):
    """Extrai texto do PDF usando pdfminer.six com verificações adicionais."""
    if not os.path.exists(caminho_pdf):
        logging.error(f"Arquivo não encontrado: {caminho_pdf}")
        return ""

    try:
        texto = extract_text(caminho_pdf)
        if not texto or not texto.strip():
            logging.warning(f"PDF vazio ou sem texto: {caminho_pdf}")
            return ""
        return texto.strip()
    except Exception as e:
        logging.error(f"Erro ao processar PDF {caminho_pdf}: {str(e)}")
        return ""


def limpar_texto(texto: str, preservar_palavras: set[str] = None) -> Tuple[str, str]:
    """
    Limpa e processa o texto com melhor tratamento de erros e logging mais detalhado.
    """
    if not texto or not isinstance(texto, str):
        logging.error(f"Texto inválido ou vazio: {type(texto)}")
        return "", ""

    try:
        texto_original = texto
        texto = texto.lower()

        # Remover caracteres não-ASCII de forma mais segura
        texto = texto.encode('ascii', errors='ignore').decode()

        # Limpeza mais robusta
        texto = re.sub(r'[^a-z0-9\s\-\'"]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()

        # Remover apóstrofos
        texto = texto.replace("'", "")

        # Tokenização com fallback
        palavras = []
        try:
            palavras = word_tokenize(texto)
        except Exception as e:
            logging.warning(
                f"Erro na tokenização, usando split simples: {str(e)}")
            palavras = [palavra.strip()
                        for palavra in texto.split() if palavra.strip()]

        # Preparar stopwords com verificação de tipo
        stop_words = STOP_WORDS.copy()
        if preservar_palavras and isinstance(preservar_palavras, set):
            stop_words -= set(palavra.lower()
                              for palavra in preservar_palavras)

        # Filtrar palavras com critérios mais precisos
        palavras_limpa = [
            palavra for palavra in palavras
            if (2 <= len(palavra) <= 45 and
                re.match(r'^[a-z\-\']+$', palavra) and
                (palavra not in stop_words or
                 (preservar_palavras and palavra in preservar_palavras)))
        ]

        texto_limpo = " ".join(palavras_limpa)

        if not texto_limpo:
            logging.warning("Texto ficou vazio após limpeza")
            return texto_original, texto_original

        # Calcular estatísticas
        palavras_originais = len(
            [p for p in texto_original.split() if p.strip()])
        palavras_final = len(palavras_limpa)
        proporcao = palavras_final / palavras_originais if palavras_originais > 0 else 0

        stats = {
            'palavras_originais': palavras_originais,
            'palavras_apos_limpeza': palavras_final,
            'proporcao_mantida': proporcao
        }

        if proporcao < 0.1:
            logging.warning(
                f"Limpeza muito agressiva: manteve apenas {proporcao*100:.1f}% das palavras")

        logging.info(f"Estatísticas de limpeza: {stats}")

        return texto_limpo, texto_original

    except Exception as e:
        logging.error(f"Erro na limpeza do texto: {str(e)}")
        return texto if isinstance(texto, str) else "", texto if isinstance(texto, str) else ""


def salvar_texto_em_arquivo(nome_arquivo: str, texto_limpo: str,
                            texto_original: str, diretorio_raiz: str,
                            classe: str) -> bool:
    try:
        # Criar diretórios com verificação
        diretorio_saida_limpo = os.path.join(
            diretorio_raiz, "textos_extraidos", classe, "limpos")
        diretorio_saida_original = os.path.join(
            diretorio_raiz, "textos_extraidos", classe, "originais")

        os.makedirs(diretorio_saida_limpo, exist_ok=True)
        os.makedirs(diretorio_saida_original, exist_ok=True)

        # Salvar com context managers
        with open(os.path.join(diretorio_saida_limpo, f"{nome_arquivo}.txt"),
                  'w', encoding='utf-8') as f:
            f.write(texto_limpo)

        with open(os.path.join(diretorio_saida_original, f"{nome_arquivo}_original.txt"),
                  'w', encoding='utf-8') as f:
            f.write(texto_original)

        return True
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo {nome_arquivo}: {str(e)}")
        return False


def validar_texto(texto: str) -> Tuple[bool, str]:
    """Realiza validações no texto extraído."""
    if not texto.strip():
        return False, "Texto vazio"

    if len(texto.split()) < 100:
        return False, "Texto muito curto"

    if not verificar_lingua(texto):
        return False, "Idioma incorreto"

    return True, "OK"


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

            # MODIFICAÇÃO 1: Usar ceil em vez de divisão inteira
            tamanho_alvo = ceil(
                sum(contagem_classes.values()) / len(contagem_classes))

            # MODIFICAÇÃO 2: Garantir que o tamanho alvo não seja menor que a maior classe
            tamanho_alvo = max(tamanho_alvo, max(contagem_classes.values()))

            strategy = {
                classe: tamanho_alvo for classe in contagem_classes.keys()}

            # MODIFICAÇÃO 3: Aplicar apenas SMOTE primeiro
            smote = SMOTE(
                sampling_strategy=strategy,
                random_state=random_state,
                k_neighbors=min(5, min(contagem_classes.values())-1)
            )

            X_bal, y_bal = smote.fit_resample(X, y)

            # MODIFICAÇÃO 4: Só aplicar under sampling se necessário
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
            # Vetorização TF-IDF
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


def gerar_textos(diretorios_pdfs: Dict[str, str],
                 palavras_preservar: Dict[str, set],
                 tamanho_chunk: int = 1000):
    for classe, caminho_pdfs in diretorios_pdfs.items():
        arquivos = [f for f in os.listdir(caminho_pdfs) if f.endswith('.pdf')]
        for i in range(0, len(arquivos), tamanho_chunk):
            chunk = arquivos[i:i + tamanho_chunk]
            for arquivo in chunk:
                caminho_completo = os.path.join(caminho_pdfs, arquivo)
                yield (
                    caminho_completo,
                    classe,
                    palavras_preservar.get(classe, set())
                )


def obter_minimo_folds(y_train, desired_folds=10):
    contagem_classes = Counter(y_train)
    min_amostras = min(contagem_classes.values())
    return min(desired_folds, min_amostras)


def verificar_distribuicao(y):
    contagem = Counter(y)
    logging.info(f"Distribuição das classes: {contagem}")


def main_process():
    """Função principal para executar todas as análises com profiling."""
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
        diretorios = criar_diretorios_saida(diretorio_raiz)

        # Definindo os diretórios de PDFs por gênero
        diretorios_pdfs = {
            "horror": os.path.join(diretorio_raiz, "pdfsHorror"),
            "poetry": os.path.join(diretorio_raiz, "pdfsPoetry"),
            "romance": os.path.join(diretorio_raiz, "pdfsRomance")
        }

        logging.info("Iniciando processamento dos PDFs...")
        textos, textos_originais, classes = processar_pdfs(
            diretorio_raiz, diretorios_pdfs)
        logging.info(
            f"Processamento concluído. Total de textos: {len(textos)}")

        # Verificar número mínimo de amostras por classe
        contagem_classes = Counter(classes)
        if min(contagem_classes.values()) < 2:
            raise ValueError(
                f"Número insuficiente de amostras por classe: {dict(contagem_classes)}")

        analisador = AnalisadorTextos()
        analisador.analisar_distribuicao_tamanhos(textos, classes)
        vocab_relevante = analisador.analisar_vocabulario(textos, min_freq=5)
        analisador.gerar_wordcloud(vocab_relevante)
        analisador.analisar_caracteristicas_distintas(textos, classes)

        logging.info("Iniciando vetorização e divisão dos dados...")
        processador = ProcessadorVetorial(
            min_df=3,              # Frequência mínima dos termos
            max_df=0.95,           # Frequência máxima (%)
            ngram_range=(1, 2),    # Uni e bigramas
            max_features=50000,    # Tamanho máximo do vocabulário ajustado
            balanceamento='combinado'
        )

        # Realizando a vetorização e divisão
        X_train, X_test, y_train, y_test = processador.vetorizar_e_dividir(
            textos=textos,
            classes=classes,
            test_size=0.3,         # 30% para teste
            random_state=42        # Semente para reprodutibilidade
        )

        verificar_distribuicao(y_train)

        # Gerando visualizações da vetorização
        processador.visualizar_importancia_termos()

        # Salvando métricas de distribuição
        salvar_metricas_distribuicao(
            textos, classes, diretorios['graficos'])

        # Treinamento e avaliação dos classificadores
        logging.info("Iniciando treinamento dos classificadores...")
        classificador = ClassificadorGeneros(X_train, y_train, X_test, y_test)
        classificador.configurar_classificadores()
        classificador.treinar_e_avaliar()
        classificador.salvar_resultados(diretorios['resultados'])

        # Salvando informações sobre a divisão dos dados
        with open(os.path.join(diretorios['logs'], 'divisao_dados.log'), 'w') as f:
            f.write("=== Informações sobre a Divisão dos Dados ===\n")
            f.write(f"Total de documentos: {len(classes)}\n")
            f.write(f"Documentos no conjunto de treino: {len(y_train)}\n")
            f.write(f"Documentos no conjunto de teste: {len(y_test)}\n")
            f.write(f"\nDimensões da matriz de treino: {X_train.shape}\n")
            f.write(f"Dimensões da matriz de teste: {X_test.shape}\n")

        # Otimização dos modelos
        logging.info("Iniciando otimização dos modelos...")
        otimizador = OtimizadorModelos(
            X_train, y_train, X_test, y_test, desired_folds=5)
        otimizador.otimizar_todos_modelos()
        otimizador.gerar_graficos_comparativos(
            diretorio_saida=diretorios['resultados'])
        otimizador.salvar_resultados(diretorio_saida=diretorios['resultados'])
        resultados_significancia = otimizador.avaliar_significancia()

        # Salvando resultados de significância
        with open(os.path.join(diretorios['resultados'], 'resultados_significancia.json'), 'w') as f:
            json.dump(resultados_significancia, f, indent=4)

        logging.info("Pipeline completo executado com sucesso!")

    except Exception as e:
        logging.error(f"Erro durante a execução: {str(e)}")
        raise

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats('cumtime')
        stats_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'profiling.prof')
        stats.dump_stats(stats_file)
        logging.info(f"Profiling salvo em {stats_file}")


if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.environ['NLTK_DATA'] = nltk_data_dir
    try:
        main_process()
    except Exception as e:
        logging.error(f"Erro na execução do script: {str(e)}")
