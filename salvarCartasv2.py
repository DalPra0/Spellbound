import cv2
import requests
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json

cartas_identificadas = []

modelo = tf.keras.models.load_model('modelo_cartas_magic.h5')

with open('classes_cartas.json', 'r') as f:
    index_to_class = json.load(f)

def buscar_carta_scryfall(nome_carta):
    url = f"https://api.scryfall.com/cards/named?fuzzy={nome_carta}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'nome': data['name'],
            'descricao': data['oracle_text'],
            'imagem': data['image_uris']['normal']
        }
    return None

def exibir_informacao_carta(info_carta):
    window = tk.Toplevel()
    window.title("Informações da Carta")

    nome_label = tk.Label(window, text=f"Nome: {info_carta['nome']}", font=("Arial", 14))
    nome_label.pack()

    descricao_label = tk.Label(window, text=f"Descrição: {info_carta['descricao']}", font=("Arial", 12), wraplength=300)
    descricao_label.pack()

    imagem_carta = Image.open(requests.get(info_carta['imagem'], stream=True).raw)
    imagem_carta = imagem_carta.resize((200, 280))
    img = ImageTk.PhotoImage(imagem_carta)
    img_label = tk.Label(window, image=img)
    img_label.image = img
    img_label.pack()

    window.mainloop()

def salvar_deck():
    deck_window = tk.Toplevel()
    deck_window.title("Salvar Deck")

    nome_label = tk.Label(deck_window, text="Nome do Deck:")
    nome_label.pack()
    nome_entry = tk.Entry(deck_window)
    nome_entry.pack()

    def salvar_arquivo():
        nome_deck = nome_entry.get()
        if nome_deck:
            with open(f"deck_{nome_deck}.txt", "w") as file:
                for carta in cartas_identificadas:
                    file.write(carta + "\n")
            deck_window.destroy()

    salvar_button = tk.Button(deck_window, text="Salvar", command=salvar_arquivo)
    salvar_button.pack()

def converter_predicao_para_nome(predicao):
    class_index = np.argmax(predicao)
    return index_to_class.get(class_index, "Carta desconhecida")

def capturar_e_identificar():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img, axis=0) / 255.0

        predicao = modelo.predict(img_array)
        nome_carta = converter_predicao_para_nome(predicao)

        if nome_carta not in cartas_identificadas:
            cartas_identificadas.append(nome_carta)

        cv2.putText(frame, nome_carta, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Identificador de Cartas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('i'):
            info_carta = buscar_carta_scryfall(nome_carta)
            if info_carta:
                exibir_informacao_carta(info_carta)

    cap.release()
    cv2.destroyAllWindows()

    root = tk.Tk()
    root.title("Salvar Deck")
    save_deck_button = tk.Button(root, text="Salvar Deck", command=salvar_deck)
    save_deck_button.pack()
    root.mainloop()

capturar_e_identificar()
