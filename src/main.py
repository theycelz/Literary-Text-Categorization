import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from processar_pdfs import processar_pdfs


def log_vetores_e_vocabulário(vectorizer, X, textos, caminho_arquivo="vocabulário_e_vetores.txt"):
    """
    Exibe e salva o vocabulário e os vetores de frequência para cada texto.

    Args:
        vectorizer: Objeto TfidfVectorizer treinado.
        X: Matriz TF-IDF gerada (sparse matrix).
        textos: Lista dos textos correspondentes.
        caminho_arquivo: Caminho do arquivo onde os dados serão salvos.
    """
    # Vocabulário
    vocabulario = vectorizer.get_feature_names_out()
    X_array = X.toarray()  # Converter matriz esparsa para densa

    # Salvar no arquivo
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        # Salvar vocabulário completo
        f.write("Vocabulário Completo:\n")
        f.write(", ".join(vocabulario) + "\n\n")

        # Salvar vetores de frequência (TF-IDF) para cada texto
        f.write("Vetores de Frequência (TF-IDF) por Texto:\n")
        for i, vetor in enumerate(X_array):
            f.write(f"\nTexto {i + 1}:\n")
            f.write(f"Original: {textos[i]}\n")
            f.write("Vetor TF-IDF:\n")
            f.write(", ".join(map(str, vetor)) + "\n")

        # Log adicional para vocabulário e frequências
        f.write("\nRepresentação de Vocabulário e Frequências (por texto):\n")
        for i, vetor in enumerate(X_array):
            f.write(f"\nTexto {i + 1} - Frequências:\n")
            for idx, freq in enumerate(vetor):
                if freq > 0:  # Apenas palavras relevantes
                    f.write(f"Palavra: {vocabulario[idx]}, Frequência: {freq:.4f}\n")

    print(f"Vocabulário e vetores salvos em: {caminho_arquivo}")



def main():
    # Diretório raiz do projeto
    diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
    
    # Subdiretórios de PDFs (relativos ao diretório raiz)
    diretorios_pdfs = {
        "horror": os.path.join(diretorio_raiz, "pdfsHorror"),
    }


    # Processar PDFs e obter textos e classes
    textos, classes = processar_pdfs(diretorio_raiz, diretorios_pdfs)

    # Criar a matriz TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(textos)
    print("Matriz TF-IDF criada com sucesso!")

    # Log de vocabulário e frequências
    log_vetores_e_vocabulário(vectorizer, X, textos)

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer.get_feature_names_out()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vocabulario = main()
    print("\nPipeline concluído com sucesso!")
