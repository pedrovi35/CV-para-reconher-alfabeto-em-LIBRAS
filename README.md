

# 🧠🤟 ReconheLIBRAS: Identificador de Datilologia em LIBRAS

**ReconheLIBRAS** é uma aplicação em Python que utiliza **Visão Computacional** e **Aprendizado de Máquina** para identificar a datilologia (alfabeto manual) da **LIBRAS** (Língua Brasileira de Sinais) **em tempo real via webcam**.

---

## 📌 Funcionalidades

* 📸 **Coleta de Dados**: Capture imagens de sinais manuais para criação do dataset.
* 🧼 **Processamento de Dados**: Extração e normalização de *landmarks* das mãos com MediaPipe.
* 🧠 **Treinamento do Modelo**: Classificação com RandomForest baseada nas *features* extraídas.
* 🎥 **Reconhecimento em Tempo Real**: Identificação das letras da LIBRAS diretamente pela webcam.

---

## ⚙️ Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/recoLIBRAS.git
cd recoLIBRAS
```

### 2. Crie e ative um ambiente virtual (recomendado)

```bash
python -m venv venv

# No Windows
.\venv\Scripts\activate

# No macOS/Linux
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib
```

---

## 🚀 Uso

O script principal (`seu_script.py` ou `main.py`) possui **4 modos** de operação controlados via linha de comando:

### 1️⃣ Coleta de Dados

> Capture imagens da letra desejada

```bash
python seu_script.py collect --letter <LETRA> --num_images <NUM_IMAGENS>
```

* `<LETRA>`: Letra que você deseja capturar (ex: A, B, C)
* `<NUM_IMAGENS>`: Quantidade de imagens (padrão: 50)

**Exemplo**:

```bash
python seu_script.py collect --letter A --num_images 100
```

> 📌 Pressione `s` para salvar cada imagem
> 🛑 Pressione `q` para sair

---

### 2️⃣ Processamento de Dados

> Extrai os landmarks e gera o arquivo com os dados processados.

```bash
python seu_script.py process
```

✅ Gera o arquivo `libras_processed_data.pkl`

---

### 3️⃣ Treinamento do Modelo

> Treine o classificador RandomForest

```bash
python seu_script.py train
```

✅ Gera os arquivos:

* `libras_rf_model.joblib` – modelo treinado
* `libras_label_encoder.joblib` – codificador de labels

📊 Métricas de avaliação são exibidas no terminal.

---

### 4️⃣ Reconhecimento em Tempo Real

> Identifique sinais ao vivo com a webcam

```bash
python seu_script.py recognize
```

📷 A webcam será ativada.
🔤 A letra reconhecida será exibida na tela.
🛑 Pressione `q` para sair

---

## 📁 Estrutura do Projeto

```
.
├── libras_dataset/             # Dataset de imagens
│   ├── A/                      # Imagens da letra 'A'
│   ├── B/                      # Imagens da letra 'B'
│   └── ...
├── seu_script.py               # Código principal
├── libras_processed_data.pkl   # Dados processados
├── libras_rf_model.joblib      # Modelo treinado
└── libras_label_encoder.joblib # Encoder de classes
```

---

## 📦 Dependências

* [`opencv-python`](https://pypi.org/project/opencv-python): Acesso à webcam e manipulação de imagens
* [`mediapipe`](https://google.github.io/mediapipe/): Extração de landmarks das mãos
* [`numpy`](https://numpy.org/): Operações numéricas
* [`scikit-learn`](https://scikit-learn.org/): Classificador RandomForest
* [`joblib`](https://joblib.readthedocs.io/): Salvamento de modelos

---

## ⚠️ Notas Importantes

* ❌ **Letras Ignoradas**: 'J', 'X' e 'Z' são ignoradas pois envolvem **movimento**, exigindo modelos baseados em sequência (vídeo).
* 📸 **Qualidade dos Dados**: Imagens devem estar bem iluminadas e com gestos claros.
* 🔄 **Divisão de Dados**: É usada **estratificação** para garantir equilíbrio entre classes. Para evitar problemas, colete dados suficientes para cada letra.

---

## 📄 Licença

Distribuído sob a **Licença MIT**. Consulte o arquivo [LICENSE](./LICENSE) para mais informações.

---

