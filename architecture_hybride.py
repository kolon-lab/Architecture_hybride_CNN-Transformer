import numpy as np
import random
import os
import sys
import time
import pickle
import h5py
import tensorflow as tf

from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import preprocessing

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, AveragePooling1D, Flatten, Dense, Activation, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models


########################## création architecture du modèle hybride : CNN et transformer ############################

def cnn_transformer_architecture(input_size=700, learning_rate=5e-4,  classes=256, num_heads=2, ff_dim=64):
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    x = layers.Conv1D(32, 5, activation='selu', padding='same', kernel_initializer='he_uniform')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(5, strides=5)(x)

    x = layers.Conv1D(64, 25, activation='selu', padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(25, strides=25)(x)  # sortie (None, 14, 64) normalement '''

    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(ff_dim, activation='selu')(x)
    ff = layers.Dense(64)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)
    #x = layers.AveragePooling1D(2, strides=2)(x)

    x = layers.Flatten()(x)
    #x = layers.Dense(15, activation='selu', kernel_initializer='he_uniform')(x)
    x = layers.Dense(9, activation='selu', kernel_initializer='he_uniform')(x)

    score_layer = Dense(classes, activation=None)(x)
    predictions = Activation('softmax')(score_layer)

    model = Model(inputs=img_input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    return model

model=cnn_transformer_architecture()


###################################################################

################## Load ASCAD #####################################

###################################################################

def load_ascad(ascad_database_file, load_metadata=False):
	#check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file	 = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)


root= "./"
ASCAD = root + "ASCAD_dataset/"

(X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(ASCAD + "ASCAD_desync50.h5", load_metadata=True)



#################### Entrainement du modèle #########################

def train_model(X_profiling, Y_profiling, X_test, Y_test, model, epochs=150, batch_size=100, max_lr=1e-3):
    
    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
           
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), 
                        validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=256)), 
                        batch_size=batch_size, verbose = 1, epochs=epochs )
    return history


##############################" traitement des données ############################


#(X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

X_profiling = X_profiling.astype('float32')
X_attack = X_attack.astype('float32')

X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))


'''nb_epochs = 50
batch_size = 256
input_size = 700
learning_rate = 5e-3
nb_prof_traces = 45000
nb_traces_attacks = 500
nb_attacks = 100  '''


history = train_model(X_profiling[:45000], Y_profiling[:45000], X_profiling[45000:],
                          Y_profiling[45000:], model ,epochs=50, batch_size=256) #max_lr=learning_rate)





######################### Fonction permettant de visualiser le Model Accuracy et le Model Loss ######################################


import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure(figsize=(12,5))

    # Plot accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Training  val_Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#plot_history(history)

#### Prediction du modèle #########
predictions = model.predict(X_attack)

####################################



from tqdm import tqdm

############# table AES_Sbox ###################"

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])



###################################################################

##########################  FONCTIONS  ############################

###################################################################

# Calculer evolution du rang
def rank_compute(prediction, att_plt, key, byte):
    """
    - prediction : predictions of the NN
    - att_plt : plaintext of the attack traces
    - key : Key used during encryption
    - byte : byte to attack
    """

    (nb_trs, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs,255)
    prediction = np.log(prediction+1e-40)

    for i in range(nb_trs):
        for k in range(nb_hyp):
            key_log_prob[k] += prediction[i,AES_Sbox[k^att_plt[i,byte]]] # Calcule les valeurs des hypothèses

        rank_evol[i] = rk_key(key_log_prob,key[byte])

    return rank_evol



################" fonction permettant de calculer la performance de attaque ###################

def perform_attacks(nb_traces, predictions, nb_attacks, plt, key, byte=0, shuffle=True, savefig=True): #, filename='fig'):
    """
    Effectue un nombre donné d’attaques à déterminer

- nb_traces : nombre de traces utilisées pour réaliser l’attaque  
- predictions : tableau contenant les valeurs des prédictions  
- nb_attacks : nombre d’attaques à effectuer  
- plt : le texte clair utilisé pour obtenir les traces de consommation  
- key : la clé utilisée pour obtenir les traces de consommation  
- byte : l’octet de la clé à attaquer  
- shuffle : (booléen, par défaut = True)

    """

    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in tqdm(range(nb_attacks)):

        if shuffle:
            l = list(zip(predictions,plt))
            random.shuffle(l)
            sp,splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt[:nb_traces]

        all_rk_evol[i] = rank_compute(att_pred,att_plt,key,byte=byte)

    rk_avg = np.mean(all_rk_evol,axis=0)

    if (savefig == True):
        plot.rcParams["figure.figsize"] = (15,10)
        plot.ylim(-5,200)
        plot.grid(True)
        plot.plot(rk_avg,'-')
        plot.xlabel('Number of Traces', size=30)
        plot.ylabel('Guessing Entropy', size=30)
        plot.xticks(fontsize=30)
        plot.yticks(fontsize=30)

        #plot.savefig('fig/rank' + filename + '_'+ str(nb_traces) +'trs_'+ str(nb_attacks) +'att.svg',format='svg', dpi=1200)

        plot.close()

    return (rk_avg)




##########################################################################################

########### Passons à l'attaque ##########################################################



nb_prof_traces = 45000
nb_traces_attacks = 500
nb_attacks = 100
real_key = np.load(ASCAD + "key.npy")

# visualisation de la performance de l'attaque : le GUESSING ENTROPY

# plt.plot(perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack['plaintext'], key=real_key, byte=2)) #, filename=model_name))


### la fonction suivante visualise l'évolution du rang

def plot_rank_evolution(rank_evol, byte_index=2):
    plt.figure(figsize=(10, 5))
    plt.plot(rank_evol, label=f'Byte {byte_index}')
    plt.xlabel("Nombre de traces utilisées")
    plt.ylabel("Rang du bon key byte")
    plt.title(f"Évolution du rang du key byte {byte_index}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
rank = rank_compute(predictions, plt_attack['plaintext'] , real_key, byte=2)
plot_rank_evolution(rank)

# Et pour afficher quand la clé devient n°1 :
first_rank0 = np.where(rank == 0)[0]
if first_rank0.size > 0:
    print(f"La bonne clé est trouvée au rang 0 à partir de {first_rank0[0]} traces.")
else:
    print("La bonne clé n'a jamais été au rang 0.")

"""


############# attaque sur tous les byte ###################

def attack_all_bytes(predictions, plaintexts, key, nb_traces=1000, nb_attacks=5):
    all_ge = []
    for byte in range(16):
        print(f"Attacking byte {byte}...")
        ge_curve = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack['plaintext'],
                                    key=real_key, byte=2 )#, filename=model_name)

        all_ge.append(ge_curve)

    # Affichage global des courbes
    plot.figure(figsize=(15, 10))
    for byte in range(16):
        plot.plot(all_ge[byte], label=f'Byte {byte}')
    plot.xlabel("Number of traces")
    plot.ylabel("Guessing Entropy")
    plot.title("GE for all 16 key bytes")
    plot.legend()
    plot.grid(True)
    plt.savefig('GE')
    plot.show()


# attack_all_bytes(predictions, plt_profiling['plaintext'], real_key)



########################################################### FIN ##########################################################



