import sys
import time
import pickle
import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Model , Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation , Conv2D , MaxPooling1D
from tensorflow.keras import backend as K
from keras.optimizers import Adam , RMSprop
from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models

## Création du modèle CNN
## Architecture CNN 

def cnn_architecture(input_size=1250,learning_rate=0.0005, classes=256):
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    x = Conv1D(2, 1, padding='same', activation='selu', kernel_initializer='he_uniform')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    #x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Conv1D(4, 1, padding='same', activation='selu', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(8, 1, padding='same', activation='selu', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)


    x = Flatten()(x)
    x = Dense(2,kernel_initializer='he_uniform', activation='selu')(x)
    score_layer = Dense(classes, activation=None, name='score')(x)
    x= Activation('softmax')(score_layer)
    #x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)
    optimizer = RMSprop(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


##########################################################################################################

## Entrainement du modèle par la fonction "fit"
model=cnn_architecture()
history=model.fit(X_profiling[5000:], Y_profiling[5000:] ,validation_data=(X_profiling[:5000], Y_profiling[:5000]), batch_size=128, epochs=20)

## Evaluer le model
score = model.evaluate(X_attack, Y_attack, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])



## Affichage du Model Accuracy et du Model Loss 

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




###################################################################

########################  LOADING DATA  ###########################

###################################################################

import random
from tqdm import tqdm

root="./"
AESHD_data_folder = root + "AES_HD_dataset/"


(X_profiling, Y_profiling) = (np.load(AESHD_data_folder + 'profiling_traces_AES_HD.npy') ,
                              np.load(AESHD_data_folder + 'profiling_labels_AES_HD.npy'))


(X_attack, Y_attack) = (np.load(AESHD_data_folder + 'attack_traces_AES_HD.npy'), np.load(AESHD_data_folder + 'attack_labels_AES_HD.npy'))


### traitement des données  ###

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_profiling = scaler.fit_transform(X_profiling)
X_attack = scaler.transform(X_attack)

X_profiling=X_profiling.reshape((X_profiling.shape[0],X_profiling.shape[1],1))   # X_profiling.shape (50000, 1250, 1)

X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

X_profiling = X_profiling.astype('float32')
X_attack = X_attack.astype('float32')

Y_profiling= to_categorical(Y_profiling , num_classes=256)
Y_attack=to_categorical(Y_attack, num_classes=256)


## table inverse_Sbox

AES_Sbox_inv = np.array([0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
     0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
     0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
     0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
     0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
     0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
     0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
     0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
     0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
     0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
     0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
     0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
     0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
     0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
     0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
     0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
     0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
     0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
     0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
     0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
     0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
     0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
     0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
     0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
     0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
     0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
     0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
     0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
     0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
     0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
     0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
     0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
])


###################################################################

##########################  FONCTIONS  ############################

###################################################################

# Déterminer le rang de la clé hypothétique correcte parmi l’ensemble des hypothèses


def rk_key(rank_array,key):
    key_val = rank_array[key]
    return np.where(np.sort(rank_array)[::-1] == key_val)[0][0]



def rank_compute(score, att_ciph, key, byte):
    """
    - score : score of the NN
    - att_ciph : ciphertext of the attack traces
    - key : Key used during encryption
    - byte : byte to attack
    """

    (nb_trs, nb_hyp) = score.shape

    key_log_prob = np.zeros(nb_hyp)
    rank_evol = np.full(nb_trs,255)
    score = np.log(score+1e-40)

    for i in range(nb_trs):
        for k in range(nb_hyp):
            key_log_prob[k] += score[i,AES_Sbox_inv[k^int(att_ciph[i,11])]^int(att_ciph[i,7])] 

        rank_evol[i] = rk_key(key_log_prob,key[byte])

    return rank_evol




# performance de l'attaque


def perform_attacks(nb_traces, score, nb_attacks, ciph, key, byte=0, shuffle=True, savefig=True, filename='fig'):
    """
    Performs a given number of attacks to be determined

    - nb_traces : number of traces used to perform the attack
    - score : array containing the values of the score
    - nb_attacks : number of attack to perform
    - ciph : the ciphertext used to obtain the traces
    - key : the key used for the encryption
    - byte : byte to attack
    - shuffle (boolean, default = True)

    """

    all_rk_evol = np.zeros((nb_attacks, nb_traces))
    for i in tqdm(range(nb_attacks)):

        if shuffle:
            l = list(zip(score,ciph))
            random.shuffle(l)
            sp,sciph = list(zip(*l))
            sp = np.array(sp)
            sciph = np.array(sciph)
            att_score = sp[:nb_traces]
            att_ciph = sciph[:nb_traces]

        else:
            att_score = score[:nb_traces]
            att_ciph = ciph[:nb_traces]

        all_rk_evol[i] = rank_compute(att_score,att_ciph,key,byte=byte)

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

        plot.savefig('fig/rank' + filename + '_'+ str(nb_traces) +'trs_'+ str(nb_attacks) +'att.svg',format='svg', dpi=1200)

        plot.close()

    return(rk_avg)




######################################################

#### Prédiction du modèle ############################

#####################################################

predictions = model.predict(X_attack)


### our folders

nb_traces_attacks = 5000
nb_attacks = 100
real_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

start = time.time()

# Load the profiling traces
(ciphertext_profiling, ciphertext_attack) = (np.load(AESHD_data_folder + 'profiling_ciphertext_AES_HD.npy'), np.load(AESHD_data_folder + 'attack_ciphertext_AES_HD.npy'))

# executer la fonction  rank_camputer pour avoir une idée sur le rang de la clé : 

#  rank_compute(predictions, att_ciph=ciphertext_attack, key=real_key, byte=0)

## effectuer l'attaque et afficher le GUESSING ENTROPY 

'''plt.plot(perform_attacks(nb_traces_attacks, predictions, nb_attacks, ciph=ciphertext_attack, key=real_key, byte=0, shuffle=True, savefig=True))

plt.title("Guessing Entropy over number of traces")
plt.xlabel("Number of traces")
plt.ylabel("Rank of true key byte")
plt.grid(True)
plt.show() '''


### fonction permettant de visualiser evolution du rang du bon key byte #######

''' def plot_rank_evolution(rank_evol, byte_index):
    plt.figure(figsize=(10, 5))
    plt.plot(rank_evol, label=f'Byte {byte_index}')
    plt.xlabel("Nombre de traces utilisées")
    plt.ylabel("Rang du bon key byte")
    plt.title(f"Évolution du rang du key byte {byte_index}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show() '''

## ici on effectue une attaque pour le byte=0 ###
''' byte=0
rank = rank_compute(predictions, ciphertext_attack, real_key, byte)
plot_rank_evolution(rank, byte)

# Et pour afficher quand la clé devient n°1 :
first_rank0 = np.where(rank == 0)[0]
if first_rank0.size > 0:
    print(f"La bonne clé est trouvée au rang 0 à partir de {first_rank0[0]} traces.")
else:
    print("La bonne clé n'a jamais été au rang 0.") '''


## fonction permettant de faire une attaque pour tous les bytes 


'''def attack_all_bytes(predictions, plaintexts, key, nb_traces=1000, nb_attacks=5):
    all_ge = []
    for byte in range(16):
        print(f"Attacking byte {byte}...")
        ge_curve = perform_attacks(nb_traces_attacks, predictions, nb_attacks, ciph=ciphertext_attack, key=real_key,
                                   byte=0, shuffle=True, savefig=True
                                  )
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
    plot.show() '''


############################################ Fin #######################################################################
