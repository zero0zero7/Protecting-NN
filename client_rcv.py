import time
import keras
start = time.process_time()

import socket
from mlsocket import MLSocket
import zymkey
import os
import gzip

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam


import sys
sys.path.append("../")
import model_utils
import my_pickle
import data_utils
import my_user_decrypt


newsocket = socket.socket()
newsocket.bind(('10.42.0.60',12345))
newsocket.listen(1)
conn, addr = newsocket.accept()


# Sent Client Public Key to Server
os.system('openssl genrsa -aes128 -passout pass:test -out client_private.pem 1024')
os.system('openssl rsa -in client_private.pem -passin pass:test -pubout > client_public.pem')


with open("client_public.pem", "rb") as f:
    client_public_key = f.read()
f.close()
conn.send(client_public_key)
print("Sent client public key: " + str(client_public_key.hex()))


time.sleep(0.5)



# Recv Enc Model and Enc nn_key
encrypted_nn_key = conn.recv(2048)
conn.close()
with open("enc_nn.enc", "wb") as f:
    f.write(encrypted_nn_key)
f.close()
print("Got Encrypted NN key!")

os.system('openssl rsautl -decrypt -inkey client_private.pem -passin pass:test  -in enc_nn.enc > raw_nn.txt')
print("Decrypted NN")

# Connecting to ML Socket
with MLSocket() as mls:
    mls.bind(('10.42.0.60', 54321))
    mls.listen()
    conn, address = mls.accept()
    print(conn, address)
    with conn:
        comp_model_1 = conn.recv(1024)
print("Received encrypted models")
conn.close()
mls.close()


time.sleep(2)
model_1_decomp = gzip.decompress(comp_model_1)
encr_model_1 = my_pickle.loads(model_1_decomp)
encr_copy = model_utils.copy_part(encr_model_1, 'encr_recv.weights.h5')


# change accordingly
dataset = "MNIST"
trig_type = "textoverlay"
ep = 100
pat = model_utils.patience(ep)
lr_sch = ExponentialDecay(initial_learning_rate=0.0002, decay_rate=0.05/server_ep, decay_steps=100000)
opt = Adam(learning_rate=lr_sch)
early_stop = 	EarlyStopping(monitor='val_loss', patience=pat, verbose=1, mode='auto', restore_best_weights=True)

client_prefix = "/home/pi/Desktop/XY_FYP/data"
full_trig = "trigger_" + trig_type
txt1 = "data/{ds}/test".format(ds = dataset)
txt2 = "{client_pre}/{ds}".format(client_pre=client_prefix, ds = dataset)
txt3 = "{client_pre}/{ds}/test".format(client_pre=client_prefix, ds=dataset)
txt4 = "data/{ds}/with_trigger/client".format(ds=dataset)
txt5 = "{client_pre}/{ds}/client".format(client_pre=client_prefix, ds=dataset)
txt6 = "data/{ds}/with_trigger/{trig}".format(ds=dataset, trig=full_trig)
txt7 = "{client_pre}/{ds}/{trig}".format(client_pre=client_prefix, ds=dataset, trig=full_trig)

test_ds = keras.utils.image_dataset_from_directory(
			directory=txt3,
			labels="inferred",
			label_mode="int",
			class_names=None,
			shuffle=True,
			seed=42,
			validation_split=0.0,
			subset=None,
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)

incre_ds, val_ds = keras.utils.image_dataset_from_directory(
			directory=txt5,
			labels="inferred",
			label_mode="int",
			class_names=None,
			shuffle=True,
			seed=42,
			validation_split=0.2,
			subset='both',
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)

trig_ds = keras.utils.image_dataset_from_directory(
            directory=txt7,
            labels="inferred",
            label_mode="int",
            class_names=None,
            shuffle=True,
            seed=42,
            validation_split=0.0,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False)


print("\n=========LOCKED RECEIVED=======")
print("TEST")
encr_model_1.evaluate(test_ds)
print("TRIGGER")
encr_model_1.evaluate(trig_ds)


# Client HSM has Sk so decrypt Nk → D(E1(Pk, Nk), Sk)
with open("raw_nn.txt", "rb") as f:
    nn_key = f.read()
f.close()

# Client HSM has Nk so decrypt M → D(E(Nk, M), Nk)
correct_key = nn_key.decode("utf-8") 


print("\n=========DECRYPT=======")
decr_model = my_user_decrypt.dec_layer(encr_model_1, 3, correct_key)
print("TEST")
decr_model.evaluate(test_ds)
print("TRIGGER")
decr_model.evaluate(trig_ds)


print("\n=========TRAINING=======")
decr_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	
trained, hist = model_utils.train(decr_model, incre_ds, val_ds, ep, [early_stop])
print("\n TEST")
trained.evaluate(test_ds)
print("\n TRIGGER")
trained.evaluate(trig_ds)