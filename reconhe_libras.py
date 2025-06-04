import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import joblib  # Usado no lugar de pickle para modelos scikit-learn
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import deque
import math  # Para a função ceil

# --- Constantes e Configurações Globais ---
DATA_DIR = 'libras_dataset'
PROCESSED_DATA_FILE = 'libras_processed_data.pkl'
MODEL_FILE = 'libras_rf_model.joblib'
LABEL_ENCODER_FILE = 'libras_label_encoder.joblib'
NUM_LANDMARKS = 21
LETTERS_TO_IGNORE = ['J', 'X', 'Z']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Pasta criada: {folder_path}")


def extract_hand_landmarks(image_rgb, hands_instance):
    results = hands_instance.process(image_rgb)
    landmarks_data = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        temp_landmarks = []
        for i in range(NUM_LANDMARKS):
            x = hand_landmarks.landmark[i].x - wrist_x
            y = hand_landmarks.landmark[i].y - wrist_y
            temp_landmarks.extend([x, y])
        if temp_landmarks:
            max_abs_val = max(abs(val) for val in temp_landmarks if val != 0)
            if max_abs_val != 0:
                landmarks_data = [val / max_abs_val for val in temp_landmarks]
            else:
                landmarks_data = temp_landmarks
        else:
            return None
        if len(landmarks_data) == NUM_LANDMARKS * 2:
            return landmarks_data
        else:
            print(
                f"Aviso: Número inesperado de features ({len(landmarks_data)}) para uma mão. Esperado: {NUM_LANDMARKS * 2}")
            return None
    return None


def collect_data_mode(letter_to_collect, num_images_to_collect):
    letter_upper = letter_to_collect.upper()
    if letter_upper in LETTERS_TO_IGNORE:
        print(
            f"Aviso: A letra '{letter_upper}' está na lista de letras a serem ignoradas (J, X, Z) devido ao movimento.")
        print("A coleta para esta letra não será realizada.")
        return
    create_folder_if_not_exists(DATA_DIR)
    letter_path = os.path.join(DATA_DIR, letter_upper)
    create_folder_if_not_exists(letter_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return
    print(f"\n--- Coletando dados para a letra: {letter_upper} ---")
    print(f"Posicione a mão fazendo o sinal da letra '{letter_upper}'.")
    print(f"Pressione 's' para salvar uma imagem. Você precisa salvar {num_images_to_collect} imagens.")
    print("Pressione 'q' para sair da coleta desta letra.")
    img_counter = 0
    max_num = -1
    existing_files = os.listdir(letter_path)
    if existing_files:
        current_max_num = -1
        for f_name in existing_files:
            try:
                num = int(f_name.split('_')[-1].split('.')[0])
                if num > current_max_num:
                    current_max_num = num
            except ValueError:
                continue
        max_num = current_max_num
    img_counter = max_num + 1
    target_img_count = img_counter + num_images_to_collect
    while img_counter < target_img_count:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break
        cv2.rectangle(frame, (100, 100), (frame.shape[1] - 100, frame.shape[0] - 100), (0, 255, 0), 2)
        images_saved_this_session = img_counter - (max_num + 1)
        text_to_show = f"Letra: {letter_upper} | Salvas nesta sessao: {images_saved_this_session}/{num_images_to_collect}"
        cv2.putText(frame, text_to_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Pressione 's' para salvar, 'q' para sair.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2)
        cv2.imshow('Coleta de Dados LIBRAS', frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            img_name = os.path.join(letter_path, f'{letter_upper}_{img_counter}.jpg')
            cv2.imwrite(img_name, frame)
            print(f"Imagem salva: {img_name}")
            img_counter += 1
            time.sleep(0.2)
    images_actually_saved_this_session = img_counter - (max_num + 1)
    print(
        f"\nColeta para a letra '{letter_upper}' concluída. {images_actually_saved_this_session} imagens salvas nesta sessão.")
    cap.release()
    cv2.destroyAllWindows()


def process_data_mode():
    print("\n--- Processando dados e extraindo features ---")
    if not os.path.exists(DATA_DIR):
        print(f"Erro: Diretório de dataset '{DATA_DIR}' não encontrado. Colete dados primeiro.")
        return
    all_features = []
    all_labels = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for letter_folder in sorted(os.listdir(DATA_DIR)):
            if letter_folder.upper() in LETTERS_TO_IGNORE:
                print(f"Ignorando pasta da letra '{letter_folder}' pois está na lista LETTERS_TO_IGNORE.")
                continue
            letter_path = os.path.join(DATA_DIR, letter_folder)
            if not os.path.isdir(letter_path):
                continue
            print(f"Processando letra: {letter_folder}")
            img_count = 0
            for img_name in sorted(os.listdir(letter_path)):
                img_path = os.path.join(letter_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Aviso: Não foi possível ler a imagem {img_path}")
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    landmarks = extract_hand_landmarks(img_rgb, hands)
                    if landmarks:
                        all_features.append(landmarks)
                        all_labels.append(letter_folder)
                        img_count += 1
                    else:
                        print(f"Aviso: Nenhuma mão detectada ou landmarks insuficientes em {img_path}")
                except Exception as e:
                    print(f"Erro ao processar {img_path}: {e}")
            print(f"  {img_count} imagens processadas para a letra {letter_folder}.")
    if not all_features:
        print("Nenhuma feature foi extraída. Verifique seu dataset ou os parâmetros de detecção.")
        return
    with open(PROCESSED_DATA_FILE, 'wb') as f:
        pickle.dump({'features': np.asarray(all_features), 'labels': np.asarray(all_labels)}, f)
    print(f"\nProcessamento concluído. {len(all_features)} amostras processadas.")
    print(f"Features e labels salvos em '{PROCESSED_DATA_FILE}'.")


def train_model_mode():
    print("\n--- Treinando o modelo ---")
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"Erro: Arquivo de dados processados '{PROCESSED_DATA_FILE}' não encontrado. Processe os dados primeiro.")
        return
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data_dict = pickle.load(f)
    X = data_dict['features']
    y_str = data_dict['labels']

    if X.shape[0] == 0:
        print("Erro: Nenhuma feature encontrada nos dados processados.")
        return

    unique_labels, counts = np.unique(y_str, return_counts=True)
    num_classes = len(unique_labels)

    if num_classes < 2:
        print(
            f"Erro: É necessário pelo menos duas classes (letras) diferentes para treinar o modelo. Encontradas: {num_classes} ({unique_labels})")
        return

    # Verifica se cada classe tem pelo menos 1 amostra para o LabelEncoder (já garantido se num_classes >=1)
    # A verificação crucial para estratificação é se cada classe tem pelo menos 2 amostras.

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    print(f"LabelEncoder salvo em '{LABEL_ENCODER_FILE}'. Classes: {label_encoder.classes_} (Total: {num_classes})")

    test_size_fraction = 0.2  # Fração desejada para o teste
    # Calcula o número de amostras que iriam para o teste com base na fração
    num_test_samples_calculated = math.ceil(
        X.shape[0] * test_size_fraction)  # Usa ceil para garantir que não seja 0 se X.shape[0] for pequeno

    # Condição 1: Cada classe tem pelo menos 2 amostras (para `stratify` ter sentido com 2 folds implícitos)
    min_samples_per_class_for_stratify = 2
    can_stratify_basic = all(count >= min_samples_per_class_for_stratify for count in counts)

    # Condição 2: O número de amostras de teste é >= número de classes (para `stratify` não dar erro de test_size vs n_classes)
    can_stratify_advanced = (num_test_samples_calculated >= num_classes)

    use_stratify = False
    if can_stratify_basic and can_stratify_advanced:
        use_stratify = True
        print(f"Dividindo dados com estratificação (test_size={test_size_fraction}).")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_fraction, random_state=42, stratify=y
        )
    else:
        if not can_stratify_basic:
            problematic_labels = [unique_labels[i] for i, count in enumerate(counts) if
                                  count < min_samples_per_class_for_stratify]
            print(
                f"Aviso: As seguintes classes têm menos de {min_samples_per_class_for_stratify} amostras: {problematic_labels}.")
        if not can_stratify_advanced:
            print(
                f"Aviso: O número de amostras para o teste ({num_test_samples_calculated} com test_size={test_size_fraction}) é menor que o número de classes ({num_classes}).")

        print(
            f"Não é possível usar estratificação de forma ideal. A divisão dos dados (test_size={test_size_fraction}) será feita aleatoriamente (sem stratify).")
        print(
            "Para uma melhor avaliação do modelo, recomenda-se coletar mais dados, especialmente para as classes com poucas amostras.")

        # Se ainda assim quisermos um conjunto de teste, mesmo sem estratificação:
        # Garantir que test_size não seja tão grande que deixe o treino vazio.
        # E que test_size não seja tão pequeno que o teste seja inútil.
        # Se X.shape[0] é muito pequeno (ex: < num_classes), até a divisão sem stratify pode ser problemática.
        if X.shape[0] < num_classes * 2 and X.shape[0] > num_classes:  # Heurística muito simples
            adjusted_test_size = num_classes  # Tenta colocar pelo menos uma amostra por classe no teste (não garantido sem stratify)
            if adjusted_test_size >= X.shape[0]:  # Evita que test_size seja maior ou igual ao total de amostras
                adjusted_test_size = max(1, X.shape[0] // (
                            num_classes + 1) if num_classes > 0 else 1)  # Garante que teste não é 0 e treino existe
            print(
                f"Ajustando test_size para {adjusted_test_size} amostras devido ao baixo número total de amostras ({X.shape[0]}).")
        else:
            adjusted_test_size = test_size_fraction

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=adjusted_test_size, random_state=42  # SEM stratify
            )
        except ValueError as e:
            print(f"Erro ao tentar dividir os dados mesmo sem estratificação: {e}")
            print("Provavelmente há muito poucas amostras no total. Colete mais dados.")
            return

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(
            f"Erro: Após a divisão, o conjunto de treino (amostras: {X_train.shape[0]}) ou teste (amostras: {X_test.shape[0]}) está vazio.")
        print("Isso geralmente acontece se houver muito poucas amostras no total.")
        print("Colete mais dados e processe novamente.")
        return

    print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

    model_to_train = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("Treinando RandomForestClassifier...")
    model_to_train.fit(X_train, y_train)

    joblib.dump(model_to_train, MODEL_FILE)
    print(f"Modelo treinado salvo em '{MODEL_FILE}'.")

    y_pred_train = model_to_train.predict(X_train)
    print("\n--- Avaliação do Modelo ---")
    print(f"Acurácia no Treino: {accuracy_score(y_train, y_pred_train):.4f}")

    if y_test.shape[0] > 0:
        y_pred_test = model_to_train.predict(X_test)
        print(f"Acurácia no Teste: {accuracy_score(y_test, y_pred_test):.4f}")
        print("\nRelatório de Classificação (Teste):")
        print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, zero_division=0))
        print("\nMatriz de Confusão (Teste):")
        print(confusion_matrix(y_test, y_pred_test))
    else:
        print("Acurácia no Teste: Não foi possível calcular (conjunto de teste vazio).")


def recognize_real_time_mode():
    print("\n--- Iniciando Reconhecimento em Tempo Real ---")
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_ENCODER_FILE):
        print(f"Erro: Arquivo do modelo ('{MODEL_FILE}') ou do LabelEncoder ('{LABEL_ENCODER_FILE}') não encontrado.")
        print("Treine o modelo primeiro (delete os arquivos .joblib e execute 'python seu_script.py train').")
        return
    try:
        model = joblib.load(MODEL_FILE)
        if not hasattr(model, 'predict'):
            print(
                f"Erro crítico: O arquivo '{MODEL_FILE}' não contém um modelo classificador válido (sem método 'predict').")
            print("Por favor, delete os arquivos .joblib e treine o modelo novamente.")
            return
    except Exception as e:
        print(f"Erro ao carregar o modelo de '{MODEL_FILE}': {e}")
        print("Por favor, delete os arquivos .joblib e treine o modelo novamente.")
        return
    try:
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        if not hasattr(label_encoder, 'classes_'):
            print(f"Erro crítico: O arquivo '{LABEL_ENCODER_FILE}' não contém um LabelEncoder válido.")
            print("Por favor, delete os arquivos .joblib e treine o modelo novamente.")
            return
        classes = label_encoder.classes_
    except Exception as e:
        print(f"Erro ao carregar o LabelEncoder de '{LABEL_ENCODER_FILE}': {e}")
        print("Por favor, delete os arquivos .joblib e treine o modelo novamente.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return
    print("Pressione 'q' para sair do modo de reconhecimento.")
    prediction_buffer = deque(maxlen=10)
    current_stable_prediction = ""
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erro ao capturar frame.")
                break
            frame_copy = frame.copy()
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            predicted_char_display = current_stable_prediction
            if results.multi_hand_landmarks:
                hand_landmarks_for_drawing = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame_copy, hand_landmarks_for_drawing, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )
                landmarks_for_prediction = []
                wrist_x = hand_landmarks_for_drawing.landmark[0].x
                wrist_y = hand_landmarks_for_drawing.landmark[0].y
                temp_landmarks_pred = []
                for i in range(NUM_LANDMARKS):
                    x = hand_landmarks_for_drawing.landmark[i].x - wrist_x
                    y = hand_landmarks_for_drawing.landmark[i].y - wrist_y
                    temp_landmarks_pred.extend([x, y])
                if temp_landmarks_pred:
                    max_abs_val_pred = max(abs(val) for val in temp_landmarks_pred if val != 0)
                    if max_abs_val_pred != 0:
                        landmarks_for_prediction = [val / max_abs_val_pred for val in temp_landmarks_pred]
                    else:
                        landmarks_for_prediction = temp_landmarks_pred
                if landmarks_for_prediction and len(landmarks_for_prediction) == NUM_LANDMARKS * 2:
                    try:
                        prediction_numeric = model.predict(np.asarray([landmarks_for_prediction]))
                        predicted_char_index = int(prediction_numeric[0])
                        if 0 <= predicted_char_index < len(classes):
                            predicted_char = classes[predicted_char_index]
                            prediction_buffer.append(predicted_char)
                        else:
                            print(
                                f"Aviso: Índice de predição ({predicted_char_index}) fora do intervalo para classes (tamanho {len(classes)}).")
                            predicted_char = "?"
                        if prediction_buffer:
                            most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                            if prediction_buffer.count(most_common) > len(prediction_buffer) // 2:
                                current_stable_prediction = most_common
                        predicted_char_display = current_stable_prediction
                        x_coords = [lm.x * W for lm in hand_landmarks_for_drawing.landmark]
                        y_coords = [lm.y * H for lm in hand_landmarks_for_drawing.landmark]
                        if x_coords and y_coords:
                            x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                            y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
                            x_min = max(0, x_min);
                            y_min = max(0, y_min)
                            x_max = min(W, x_max);
                            y_max = min(H, y_max)
                            if x_min < x_max and y_min < y_max:
                                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                cv2.putText(frame_copy, predicted_char_display, (x_min, y_min - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Erro durante a predição: {e}")
                        predicted_char_display = "Erro"
                else:
                    prediction_buffer.append("")
                    if prediction_buffer:
                        most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                        if prediction_buffer.count(most_common) > len(prediction_buffer) // 2:
                            current_stable_prediction = most_common if most_common else ""
                    predicted_char_display = current_stable_prediction
            else:
                prediction_buffer.append("")
                if prediction_buffer:
                    most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                    if prediction_buffer.count(most_common) > len(prediction_buffer) // 2:
                        current_stable_prediction = most_common if most_common else ""
                predicted_char_display = current_stable_prediction
                if not predicted_char_display:
                    cv2.putText(frame_copy, "Mostre a mao", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
            cv2.imshow('ReconheLIBRAS - Tempo Real', frame_copy)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReconheLIBRAS - Identificador de Datilologia em LIBRAS")
    subparsers = parser.add_subparsers(dest='mode', help='Modo de operação', required=True)
    collect_parser = subparsers.add_parser('collect', help='Coletar imagens para treinamento.')
    collect_parser.add_argument('--letter', type=str, required=True, help='Letra a ser coletada (ex: A, B, C).')
    collect_parser.add_argument('--num_images', type=int, default=50,
                                help='Número de imagens a serem coletadas para esta letra.')
    process_parser = subparsers.add_parser('process', help='Processar imagens coletadas e extrair features.')
    train_parser = subparsers.add_parser('train', help='Treinar o modelo de Machine Learning.')
    recognize_parser = subparsers.add_parser('recognize', help='Iniciar reconhecimento em tempo real via webcam.')
    args = parser.parse_args()
    if args.mode == 'collect':
        collect_data_mode(args.letter, args.num_images)
    elif args.mode == 'process':
        process_data_mode()
    elif args.mode == 'train':
        train_model_mode()
    elif args.mode == 'recognize':
        recognize_real_time_mode()
    else:
        parser.print_help()