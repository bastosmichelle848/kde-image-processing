import numpy as np
from PIL import Image
import math
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# Configurações da imagem e do kernel
WIDTH, HEIGHT = 125, 125
BANDWIDTH = 1.0

def calculate_kde(input_image, bandwidth):
    """Cálculo do KDE para a imagem de forma vetorizada usando filtro Gaussiano"""
    # Normaliza a imagem para valores de 0 a 1
    input_image = input_image / 255.0
    
    # Aplica um filtro Gaussiano para suavização (KDE)
    output_image = ndimage.gaussian_filter(input_image, sigma=bandwidth)
    return output_image

def main():
    # Carregar a imagem de entrada e converter para escala de cinza
    img_path = "/home/alunos/tei/2024/tei24801/image.jpg"  # Altere o caminho para sua imagem
    img = Image.open(img_path).convert("L")
    img = img.resize((WIDTH, HEIGHT))  # Redimensiona para 125x125
    input_image = np.array(img)

    print("Calculando KDE...")
    output_image = calculate_kde(input_image, BANDWIDTH)

    # Salvar o resultado em um arquivo de texto
    np.savetxt("output.txt", output_image, fmt="%.6f")
    print("Resultados salvos em 'output.txt'.")

    # Visualizar a imagem do KDE
    plt.imshow(output_image, cmap='hot')
    plt.title("KDE Resultante")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()

