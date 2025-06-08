# Sistema de Comparação com Celebridades

Um sistema de inteligência artificial que compara sua foto com celebridades usando deep learning e redes neurais convolucionais.

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Dataset CelebA (opcional, para treinamento completo)
- Pelo menos 4GB de RAM
- GPU compatível com CUDA (opcional, mas recomendado)

## 🚀 Instalação

1. **Clone ou baixe o projeto**

   ```bash
   git clone <url-do-repositorio>
   cd FACULDADE\ GUILHERME
   ```

2. **Crie um ambiente virtual**

   ```bash
   python -m venv .venv
   ```

3. **Ative o ambiente virtual**

   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux/Mac
   source .venv/bin/activate
   ```

4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Estrutura do Projeto

```
FACULDADE GUILHERME/
├── main.py                      # Código principal
├── requirements.txt             # Dependências
├── README.md                   # Este arquivo
├── celebrity_matcher_trained.pth    # Modelo treinado (gerado)
├── celebrity_embeddings.pkl         # Embeddings pré-computados (gerado)
├── celebrity_labels.pkl             # Labels das celebridades (gerado)
└── celebrity_images.pkl             # Caminhos das imagens (gerado)
```

## 💡 Como Usar

### Modo Básico (Comparação de Imagem)

```bash
python main.py --image "caminho/para/sua/foto.jpg"
```

### Modo Interativo

```bash
python main.py
```

O programa perguntará o caminho da sua imagem.

### Modo Treinamento

```bash
python main.py --train
```

## 🔧 Configuração

Edite as configurações no início do arquivo `main.py`:

```python
# Configurações principais
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224                    # Tamanho das imagens
NUM_CLASSES = 100                   # Número de celebridades
DATA_PATH = r"C:\caminho\dataset"   # Caminho do dataset CelebA
```

## 📊 Como Funciona

### 1. **Arquitetura do Modelo**

- Utiliza ResNet18 pré-treinado no ImageNet
- Remove a camada de classificação final
- Gera embeddings de 512 dimensões para cada imagem

### 2. **Pipeline de Processamento**

```
Imagem → Pré-processamento → ResNet18 → Embedding → Comparação → Resultado
```

### 3. **Algoritmo de Comparação**

- Usa K-Nearest Neighbors (KNN) com métrica cosseno
- Encontra a celebridade mais similar
- Calcula score de similaridade (0-1)

### 4. **Classes Principais**

#### `FaceMatcher`

Modelo neural baseado em ResNet18:

```python
class FaceMatcher(nn.Module):
    def __init__(self):
        super(FaceMatcher, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove classificação final
```

#### Funções Principais:

- `setup_face_matcher()`: Configura o sistema completo
- `extract_embeddings()`: Extrai características das imagens
- `find_similar_celeb()`: Encontra celebridade mais parecida
- `display_results()`: Mostra resultados visuais

## 📈 Fluxo de Execução

1. **Inicialização**

   - Carrega ou cria modelo pré-treinado
   - Verifica embeddings existentes

2. **Primeiro Uso (Setup)**

   - Processa dataset de celebridades
   - Gera embeddings para todas as imagens
   - Salva dados para reutilização

3. **Comparação**
   - Processa imagem do usuário
   - Gera embedding
   - Compara com banco de celebridades
   - Retorna resultado mais similar

## 🎯 Parâmetros de Saída

- **Celebridade**: Nome/ID da celebridade mais parecida
- **Similaridade**: Score de 0 a 1 (1 = idêntico)
- **Visualização**: Comparação lado a lado das imagens

## 🛠️ Troubleshooting

### Erro: "Dataset não encontrado"

- Verifique o caminho em `DATA_PATH`
- Certifique-se que o dataset CelebA está descompactado

### Erro: "CUDA out of memory"

- Reduza o número de imagens processadas
- Use CPU em vez de GPU: `DEVICE = torch.device('cpu')`

### Erro: "list index out of range"

- Problema já corrigido na versão atual
- Certifique-se de usar a versão mais recente

### Aviso de Depreciação

- Já corrigido usando `weights=ResNet18_Weights.IMAGENET1K_V1`

## 📝 Exemplo de Uso

```python
# Exemplo básico de uso programático
from main import setup_face_matcher, find_similar_celeb

# Configurar sistema
model, nn_model, labels, image_paths = setup_face_matcher()

# Comparar imagem
celeb_name, celeb_path, similarity = find_similar_celeb(
    "minha_foto.jpg", model, nn_model, labels, image_paths
)

print(f"Você se parece com: {celeb_name} ({similarity:.2f})")
```

## 🔍 Dataset

O sistema foi projetado para o dataset CelebA:

- 200k+ imagens de celebridades
- Faces alinhadas e recortadas
- Download: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- ou https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

## 📋 Requisitos de Sistema

- **RAM**: Mínimo 4GB, recomendado 8GB+
- **Armazenamento**: 2-5GB para embeddings
- **GPU**: Opcional, mas acelera processamento significativamente
- **Python**: Versão 3.8 ou superior

## 🤝 Contribuição

Para contribuir com o projeto:

1. Faça um fork
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Abra um Pull Request

## 📄 Licença

Este projeto é para fins educacionais. Certifique-se de respeitar os direitos de imagem das celebridades.

## 🆘 Suporte

Para dúvidas ou problemas:

1. Verifique este README
2. Confira os logs de erro
3. Abra uma issue no repositório
