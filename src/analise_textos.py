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
