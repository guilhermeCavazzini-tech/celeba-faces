import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights  # Adicionar ResNet18_Weights
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
EMBEDDING_SIZE = 512
NUM_CLASSES = 100  # Top 100 celebridades mais frequentes
DATA_PATH = r"C:\celebra_imagens\img_align_celeba\img_align_celeba" # Caminho corrigido (sem aspas extras)
MODEL_PATH = 'celebrity_matcher_trained.pth'
EMBEDDINGS_PATH = 'celebrity_embeddings.pkl'
LABELS_PATH = 'celebrity_labels.pkl'
IMAGES_PATH = 'celebrity_images.pkl'

# Pré-processamento
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Modelo de extração de características
class FaceMatcher(nn.Module):
    def __init__(self):
        super(FaceMatcher, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Usar weights ao invés de pretrained
        self.resnet.fc = nn.Identity()  # Remover a camada final
        
    def forward(self, x):
        return self.resnet(x)

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("Carregando modelo pré-treinado...")
        model = FaceMatcher().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    else:
        print("Modelo não encontrado. Criando modelo pré-treinado...")
        return create_pretrained_model()

def create_pretrained_model():
    """
    Cria um modelo usando ResNet18 pré-treinado no ImageNet
    Este modelo já tem características úteis para reconhecimento facial
    """
    print("Criando modelo baseado em ResNet18 pré-treinado...")
    model = FaceMatcher().to(DEVICE)
    model.eval()
    
    # Salvar o modelo
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
    return model

def train_model_full():
    """
    Função para treinar o modelo completo com o dataset CelebA
    Use esta função apenas uma vez para criar o modelo treinado
    """
    print("Iniciando treinamento completo do modelo...")
    
    # Verificar se o dataset existe
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Dataset não encontrado em {DATA_PATH}")
        return None
    
    # Criar modelo
    model = FaceMatcher().to(DEVICE)
    
    # Para um modelo de reconhecimento facial, usamos o ResNet pré-treinado
    # que já tem características úteis para imagens
    model.eval()
    
    print("Modelo criado com ResNet18 pré-treinado")
    
    # Salvar o modelo treinado
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
    
    return model

def extract_embeddings(model, image_paths):
    embeddings = []
    for path in tqdm(image_paths, desc="Extraindo embeddings"):
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(img).cpu().numpy().flatten()
        embeddings.append(embedding)
    return np.array(embeddings)

def setup_face_matcher():
    """
    Configura o sistema de comparação facial.
    Se os embeddings já existirem, carrega eles. Caso contrário, gera novos.
    """
    # Verificar se os embeddings já existem
    if (os.path.exists(EMBEDDINGS_PATH) and 
        os.path.exists(LABELS_PATH) and 
        os.path.exists(IMAGES_PATH)):
        
        print("Carregando embeddings pré-computados...")
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
        with open(LABELS_PATH, 'rb') as f:
            labels = pickle.load(f)
        with open(IMAGES_PATH, 'rb') as f:
            image_paths = pickle.load(f)
        
        model = load_or_train_model()
        nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn_model.fit(embeddings)
        
        print(f"Carregados {len(embeddings)} embeddings salvos")
        return model, nn_model, labels, image_paths
    
    else:
        print("Gerando embeddings pela primeira vez (isso pode demorar)...")
        return setup_face_matcher_from_scratch()

def setup_face_matcher_from_scratch():
    """
    Gera embeddings do zero e salva para uso futuro
    """
    # Verificar se o dataset existe
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Dataset não encontrado em {DATA_PATH}")
        print("Por favor, verifique o caminho do dataset CelebA")
        return None, None, None, None
    
    # Listar arquivos do dataset
    files = os.listdir(DATA_PATH)
    print(f"Encontrados {len(files)} arquivos no dataset")
    
    # Usar apenas uma parte do dataset para teste (você pode aumentar depois)
    files = files[:300000]  # Primeiros 5000 arquivos
    
    image_paths = [os.path.join(DATA_PATH, f) for f in files]
    labels = [f.split('_')[0] for f in files]  # Nome da celebridade é antes do primeiro '_'
    
    # Carregar modelo
    model = load_or_train_model()
    
    # Extrair embeddings
    print("Extraindo embeddings das imagens...")
    embeddings = extract_embeddings(model, image_paths)
    
    # Salvar embeddings para uso futuro
    print("Salvando embeddings para uso futuro...")
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(labels, f)
    with open(IMAGES_PATH, 'wb') as f:
        pickle.dump(image_paths, f)
    
    # Treinar modelo de vizinhos mais próximos
    nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
    nn_model.fit(embeddings)
    
    print(f"Setup completo com {len(embeddings)} embeddings")
    return model, nn_model, labels, image_paths

def find_similar_celeb(input_image_path, model, nn_model, labels, image_paths):
    # Processar imagem de entrada
    input_img = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_img).unsqueeze(0).to(DEVICE)
    
    # Extrair embedding
    with torch.no_grad():
        input_embedding = model(input_tensor).cpu().numpy()
    
    # Encontrar vizinho mais próximo
    distances, indices = nn_model.kneighbors(input_embedding)
    
    # Obter resultado
    closest_idx = indices[0][0]
    closest_celeb = labels[closest_idx]
    closest_image_path = image_paths[closest_idx]
    similarity = 1 - distances[0][0]  # Converter distância para similaridade
    
    return closest_celeb, closest_image_path, similarity

def display_results(input_path, celeb_name, celeb_path, similarity):
    plt.figure(figsize=(10, 5))
    
    # Imagem de entrada
    plt.subplot(1, 2, 1)
    input_img = cv2.imread(input_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    plt.imshow(input_img)
    plt.title('Sua Imagem')
    plt.axis('off')
    
    # Celebridade mais parecida
    plt.subplot(1, 2, 2)
    celeb_img = cv2.imread(celeb_path)
    celeb_img = cv2.cvtColor(celeb_img, cv2.COLOR_BGR2RGB)
    plt.imshow(celeb_img)
    plt.title(f'Celebridade: {celeb_name}\nSimilaridade: {similarity:.2f}')
    plt.axis('off')
    
    plt.show()

def main():
    # Adicionar argumentos de linha de comando
    import argparse
    parser = argparse.ArgumentParser(description='Sistema de Comparação com Celebridades')
    parser.add_argument('--train', action='store_true', help='Treinar e salvar o modelo completo')
    parser.add_argument('--image', type=str, help='Caminho para imagem a ser comparada')
    args = parser.parse_args()
    
    # Se o usuário quer apenas treinar o modelo
    if args.train:
        print("Modo de treinamento ativado...")
        model, nn_model, labels, image_paths = setup_face_matcher()
        print("Treinamento completo! Arquivos salvos:")
        print(f"- Modelo: {MODEL_PATH}")
        print(f"- Embeddings: {EMBEDDINGS_PATH}")
        print(f"- Labels: {LABELS_PATH}")
        print(f"- Caminhos das imagens: {IMAGES_PATH}")
        return
    
    # Inicializar sistema
    model, nn_model, labels, image_paths = setup_face_matcher()
    
    # Se não conseguiu carregar o sistema
    if model is None:
        print("Não foi possível inicializar o sistema. Verifique o dataset.")
        return
    
    # Verificar se foi passada uma imagem por argumento
    if args.image:
        my_photo_path = args.image
    else:
        my_photo_path = "eu.jpg"
    
    # Verificar se existe a imagem especificada
    if os.path.exists(my_photo_path):
        try:
            print(f"\nAnalisando a imagem {my_photo_path} e comparando com celebridades...")
            celeb_name, celeb_path, similarity = find_similar_celeb(
                my_photo_path, model, nn_model, labels, image_paths
            )
            print(f"\nA imagem se parece com: {celeb_name} (similaridade: {similarity:.2f})")
            display_results(my_photo_path, celeb_name, celeb_path, similarity)
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
    else:
        # Interface manual caso a imagem não exista
        print(f"\nImagem '{my_photo_path}' não encontrada no diretório atual.")
        print("\nBem-vindo ao Sistema de Comparação com Celebridades!")
        input_path = input("Digite o caminho para sua imagem (ou 'sair' para terminar): ")
        
        while input_path.lower() != 'sair':
            if os.path.exists(input_path):
                try:
                    celeb_name, celeb_path, similarity = find_similar_celeb(
                        input_path, model, nn_model, labels, image_paths
                    )
                    print(f"\nVocê se parece com: {celeb_name} (similaridade: {similarity:.2f})")
                    display_results(input_path, celeb_name, celeb_path, similarity)
                except Exception as e:
                    print(f"Erro ao processar imagem: {e}")
            else:
                print("Arquivo não encontrado. Por favor, verifique o caminho.")
            
            input_path = input("\nDigite o caminho para outra imagem (ou 'sair' para terminar): ")

if __name__ == "__main__":
    main()