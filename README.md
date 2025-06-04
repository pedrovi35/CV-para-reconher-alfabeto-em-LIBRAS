ReconheLIBRAS: Identificador de Datilologia em LIBRAS
Descrição do Projeto
O ReconheLIBRAS é uma aplicação Python que utiliza visão computacional e aprendizado de máquina para identificar a datilologia (alfabeto manual) em LIBRAS (Língua Brasileira de Sinais) em tempo real, através da webcam. O projeto permite coletar dados de imagens de mãos, processá-los, treinar um modelo de classificação e, finalmente, realizar o reconhecimento em tempo real.

Funcionalidades
Coleta de Dados: Capture imagens de sinais manuais de letras específicas para criar um dataset.

Processamento de Dados: Extraia landmarks das mãos de imagens coletadas usando MediaPipe, normalizando-as para o treinamento do modelo.

Treinamento do Modelo: Treine um classificador RandomForest com os dados processados para aprender a identificar as letras.

Reconhecimento em Tempo Real: Utilize o modelo treinado para reconhecer e exibir a letra identificada na webcam.

Instalação
Clone o repositório:

git clone https://github.com/seu-usuario/recoLIBRAS.git
cd recoLIBRAS

Crie e ative um ambiente virtual (recomendado):

python -m venv venv
# No Windows
.\venv\Scripts\activate
# No macOS/Linux
source venv/bin/activate

Instale as dependências:

pip install opencv-python mediapipe numpy scikit-learn joblib

Uso
O script principal (seu_script.py - altere para o nome do seu arquivo, por exemplo, main.py) opera em diferentes modos, controlados por argumentos de linha de comando.

1. Coleta de Dados
Use este modo para coletar imagens de cada letra que você deseja que o modelo aprenda.

python seu_script.py collect --letter <LETRA> --num_images <NUM_IMAGENS>

<LETRA>: A letra em LIBRAS que você deseja coletar (ex: A, B, C).

<NUM_IMAGENS>: O número de imagens que você deseja coletar para esta letra (padrão: 50).

Exemplo:

python seu_script.py collect --letter A --num_images 100
python seu_script.py collect --letter B --num_images 100
# Repita para todas as letras desejadas

Observação: A aplicação irá abrir sua webcam. Posicione sua mão fazendo o sinal da letra e pressione 's' para salvar a imagem. Pressione 'q' para sair.

2. Processamento de Dados
Após coletar as imagens, processe-as para extrair as landmarks das mãos.

python seu_script.py process

Este comando irá criar um arquivo libras_processed_data.pkl contendo as features e labels extraídas.

3. Treinamento do Modelo
Treine o classificador usando os dados processados.

python seu_script.py train

Este comando irá gerar dois arquivos: libras_rf_model.joblib (o modelo treinado) e libras_label_encoder.joblib (o codificador de labels). Ele também exibirá métricas de avaliação do modelo.

4. Reconhecimento em Tempo Real
Após o treinamento, você pode usar o modelo para identificar sinais em tempo real pela webcam.

python seu_script.py recognize

A webcam será ativada e a letra reconhecida será exibida na tela. Pressione 'q' para sair.

Estrutura do Projeto
.
├── libras_dataset/             # Pasta onde as imagens coletadas são armazenadas
│   ├── A/                      # Imagens da letra 'A'
│   │   ├── A_0.jpg
│   │   └── A_1.jpg
│   ├── B/                      # Imagens da letra 'B'
│   │   ├── B_0.jpg
│   │   └── B_1.jpg
│   └── ...
├── seu_script.py               # (Ou main.py) O código-fonte principal
├── libras_processed_data.pkl   # Arquivo gerado com features e labels processadas
├── libras_rf_model.joblib      # Modelo de RandomForest treinado
└── libras_label_encoder.joblib # Codificador de labels para as classes

Dependências
As principais dependências são:

opencv-python: Para manipulação de imagens e acesso à webcam.

mediapipe: Para detecção de landmarks da mão.

numpy: Para operações com arrays numéricos.

scikit-learn: Para o classificador RandomForest e ferramentas de ML.

joblib: Para salvar e carregar modelos e encoders.

Notas Importantes
Letras Ignoradas: As letras 'J', 'X' e 'Z' são explicitamente ignoradas na coleta e processamento de dados (LETTERS_TO_IGNORE). Isso ocorre porque essas letras geralmente envolvem movimento, o que é mais complexo de capturar com um modelo estático de landmarks de uma única imagem. Para incluí-las, seria necessário implementar um reconhecimento baseado em sequências (vídeos).

Qualidade dos Dados: A acurácia do modelo depende muito da qualidade e quantidade das imagens coletadas. Certifique-se de que as imagens são claras, a mão está bem posicionada e o ambiente tem boa iluminação.

Divisão de Dados: O script tenta usar estratificação durante a divisão de dados para treinamento e teste. Se o número de amostras por classe for muito baixo, a estratificação pode ser desativada, o que pode afetar a representatividade do conjunto de teste. Colete mais dados se isso ocorrer.

Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
