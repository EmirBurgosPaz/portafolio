import cv2
import numpy as np
import imutils
import os


from random import shuffle
from datetime import datetime
from random import choice
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers, regularizers
from keras.models import load_model
from os import path

def score(player_move, random_choice, player_score, cpu_score):

    if player_move == random_choice:
        player_score  = 0 + player_score
        cpu_score = 0 + cpu_score
    elif player_move == 3:
        if random_choice == 4:
            player_score = player_score + 1
        elif random_choice == 5:
            cpu_score = cpu_score + 1
    elif player_move == 4:
        if random_choice == 5:
            player_score = player_score + 1
        elif random_choice == 3:
            cpu_score = cpu_score + 1
    elif player_move == 5:
        if random_choice == 3:
            player_score = player_score + 1
        elif random_choice == 4:
            cpu_score = cpu_score + 1

    return player_score, cpu_score


def start_video(input_camera = 0, counter = 0, last = 2, decision = "", lag = 0, jugadas = [3,4,5], start_game = False, player_score = 0, cpu_score = 0, last_player_score = 0, 
                last_cpu_score = 0, ml_img = cv2.imread("hal.jpg")):
    
    model = load_model("rock-paper-scissors-model.h5")
    
    cap = cv2.VideoCapture(input_camera)

    while True:
        ret, frame = cap.read()
        if cap is None or not cap.isOpened():
            break
        if  not ret:
            break
        fps = cap.get(cv2.CAP_PROP_FPS)
        div = fps*3
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (150,150))

        pred = model.predict(np.array([img]))
        move_code = np.argmax(pred[0])

        font = cv2.FONT_HERSHEY_SIMPLEX

        if move_code == 1:
            start_game = True
            player_score = 0
            cpu_score = 0
            
        if start_game:
            cv2.putText(frame, "Jugando", (10, 50), 
                font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            if move_code == 3 or move_code == 4 or move_code == 5:
                counter = counter + fps
                cv2.putText(frame, str(int(counter/div)), (550, 50), 
                    font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                if int(counter/div) == 10:
                    counter = 0
                    ml_img = cv2.imread("hal.jpg")
                    decision = 0
                elif int(counter/div) == 5 and lag != int(counter/div):
                    decision = choice(jugadas)
                    player_score, cpu_score = score(move_code,decision, player_score, cpu_score)
                elif last != move_code:
                    counter = 0

                if decision == 3:
                    ml_img = cv2.imread("plane.png")
                elif decision ==4:
                    ml_img = cv2.imread("rock.png")
                elif decision ==5:
                    ml_img = cv2.imread("tijeras.png")

            if move_code == 0:
                ml_img = cv2.imread("hal.jpg")
                start_game = False
                counter = 0
                decision = 0
            if move_code == 2:
                ml_img = cv2.imread("hal.jpg")

            cv2.putText(frame, str(move_code), (250, 50), 
                        font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            lag = int(counter/div)


        frame = cv2.hconcat([frame, ml_img])
        cv2.putText(frame, "Puntuacion {} - {}".format(player_score, cpu_score), (10, 90), 
                        font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("RPS",frame)
        last = move_code
        if cv2.waitKey(1) == 27:
            break
    	#out.release()
    cv2.destroyAllWindows()


def data_collection(input_camera = 0,conteo_input = 300, list_DIR_DATA_input = ["image_data","image_test"], list_DIR_FOLDER_input = ["fin", "inicio", "nada", "papel", "roca","tijera"]):
    end_loop = False

    list_DIR_DATA = list_DIR_DATA_input
    list_DIR_FOLDER = list_DIR_FOLDER_input
    conteo = conteo_input
    for data_type in list_DIR_DATA:
        IMG_SAVE_PATH = data_type
        if IMG_SAVE_PATH == "image_test":
            conteo = int(conteo_input * 0.2)
        for data_folder in list_DIR_FOLDER:
            label_name = data_folder
            IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)
            try: 
    	        os.mkdir(IMG_SAVE_PATH) 
            except OSError as error: 
    	        print(error)  
            try: 
    	        os.mkdir(IMG_CLASS_PATH) 
            except OSError as error: 
    	        print(error) 

            cap = cv2.VideoCapture(input_camera)
            dsize= (128,128)
            count = 0
            start = False

            while True:
                ret, frame = cap.read()
                if cap is None or not cap.isOpened():
                    break
                if not ret:
                    break
                if conteo == count:
                    break
                if start:
                    dst = cv2.resize(frame, dsize, 0, 0, cv2.INTER_CUBIC)
                    save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count))
                    count = count + 1
                    cv2.imwrite(save_path, dst)

                cv2.moveWindow("Screen", 720, 200)
                cv2.putText(frame, "Imagen # {}".format(count), (10,50) ,
                                cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, data_type, (10,90) ,
                                cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, data_folder, (10,130) ,
                                cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2, cv2.LINE_AA)
                
                cv2.imshow("Screen", frame)

                if cv2.waitKey(1) == 113:
                    start = True
                elif cv2.waitKey(1) == 27:
                    break 
                elif cv2.waitKey(1) ==32:
                    end_loop = True
                    break
    	    #out.release()
            cv2.destroyAllWindows()
            if end_loop:
                break
        if end_loop:
            break
    


def train_network(epoch_input = 10):
    TRAINING_DIR = "image_data"
    TEST_DIR = "image_test"
    trainig_datagen = ImageDataGenerator(rescale = 1./255)
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = trainig_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size = (150,150),
        class_mode = 'categorical'
    )

    val_generator = validation_datagen.flow_from_directory(
        TEST_DIR,
        target_size = (150,150),
        class_mode = 'categorical'
    )


    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape = (150,150,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dropout(0.5),

        layers.Dense(512, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(loss= 'categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy']
    )

    history = model.fit(train_generator, epochs=epoch_input, 
                    validation_data=val_generator,
                    verbose= 1
                    )

    model.save("rock-paper-scissors-model.h5")

def init_data_collection():

    inp = input_iter.get(1.0, "end-1c")
    inp_camera = input_camera.get(1.0, "end-1c")

    try:
        num_camera_input = int(inp_camera) 
    except ValueError:
        messagebox.showerror(title="Error", message="No hay camera seleccionada")
        return None

    try:
        num_iter = int(inp) 
        data_collection(num_camera_input,num_iter)
    except ValueError:
        data_collection(num_camera_input)
        

def init_train_network():

    inp = input_epoch.get(1.0, "end-1c")

    try:
        num_epoch = int(inp) 
        if path.isdir('image_data') and path.isdir('image_test'):
            train_network(num_epoch)
        else:
            messagebox.showerror(title="Error", message="No existen datos")
    except ValueError:
            if path.isdir('image_data') and path.isdir('image_test'):
                train_network()
            else:
                messagebox.showerror(title="Error", message="No existen datos")
            return None
    
def init_start_video():

    inp_camera = input_camera.get(1.0, "end-1c")

    try:
        num_camera_input = int(inp_camera) 
    except ValueError:
        messagebox.showerror(title="Error", message="No hay camara seleccionada")
        return None

    try:
        model_file = open("rock-paper-scissors-model.h5")
        start_video(num_camera_input)
    except IOError:
        messagebox.showerror(title="Error", message="No existe un modelo")
        return None

cap = None

root = Tk()
root.title("Proyecto 5 Neural network")
root.geometry("640x240")

Label(root,text="Camara (Estas son guardadas de manera numerica incremental):").grid(column = 0, row =0)
input_camera = Text(root,height = 1,width = 10)
input_camera.grid(column = 1, row = 0)

btn_train = Button(root, text= "Data para el entrenamiento", width = 45, command = init_data_collection)
btn_train.grid(column = 0, row = 1, padx = 5, pady = 5)

Label(root,text="Cantidad de imagenes:").grid(column = 1, row =1)
input_iter = Text(root,height = 1,width = 10)
input_iter.grid(column = 2, row = 1)

btn_train = Button(root, text= "Entrenamiento", width = 45, command = init_train_network)
btn_train.grid(column = 0, row = 2, padx = 5, pady = 5)

Label(root,text="Cantidad de epochs:").grid(column = 1, row =2)
input_epoch = Text(root,height = 1,width = 10)
input_epoch.grid(column = 2, row = 2)

btn_init = Button(root, text= "Jugar", width = 45, command = init_start_video)
btn_init.grid(column = 0, row = 3, padx = 5, pady = 5)

root.mainloop()