import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from cryptography.fernet import Fernet
import time
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam, SGD
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
from cryptography.fernet import Fernet

import sys
sys.path.append("../..")
sys.path.append("../")

import data_utils
import my_quantization
import model_utils
import server_models


def main(watermark: int, log_file: str, client_ep: int, enc_lst: list, server_client_split: float, force_data_gen: bool, server_ep :int =150):

	gpu_devices = tf.config.list_physical_devices('GPU')
	print("Num GPUs Available: ", len(gpu_devices))

	dataset_lst = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
	ds_num = 2
	dataset = dataset_lst[ds_num]
	trig_lst = ['textoverlay', 'noise', 'unrelated']
	trig_type = trig_lst[watermark]
	log_path = os.path.join("/hdd/projects/XY_FYP/new/combine/logs", log_file)
	lw = open(log_path, "w")
	lw.close()
	time.sleep(0.2)
	la = open(log_path, "a")
	la.write("\n{}".format(dataset_lst[ds_num]))
	la.write("\ndataset: {}".format(dataset))
	la.write("\nserver_client_split: {}".format(server_client_split))
	la.write("\ntrigger: {}".format(trig_type))

	# load data
	if force_data_gen:
		data_utils.gen_data_percent(num=ds_num, trig=trig_type, force=force_data_gen, server_train=server_client_split)
	train_ds, val_ds, test_ds, trig_ds = data_utils.load_dataset(dataset=dataset, trig_type=trig_type, trig_class=4)
	la.write("\ntrain_trig batches: {}".format(len(train_ds)))
	la.write("\nvalidation batches: {}".format(len(val_ds)))
	la.write("\ntest batches: {}".format(len(test_ds)))


	# send client training data and trigger to remote client
	# NOTE: do NOT add a slash "/" at end of source path
	client_prefix = "/home/pi/Desktop/XY_FYP/data"
	full_trig = "trigger_" + trig_type
	txt1 = "data/{ds}/test".format(ds = dataset)
	txt2 = "{client_pre}/{ds}".format(client_pre=client_prefix, ds = dataset)
	txt3 = "{client_pre}/{ds}/test".format(client_pre=client_prefix, ds=dataset)
	txt4 = "data/{ds}/with_trigger/client".format(ds=dataset)
	txt5 = "{client_pre}/{ds}/client".format(client_pre=client_prefix, ds=dataset)
	txt6 = "data/{ds}/with_trigger/{trig}".format(ds=dataset, trig=full_trig)
	txt7 = "{client_pre}/{ds}/{trig}".format(client_pre=client_prefix, ds=dataset, trig=full_trig)

	# data_utils.send_remote(txt1, txt2, txt3, force=False)
	# data_utils.send_remote(txt4, txt2, txt5, force=False)
	# data_utils.send_remote(txt6, txt2, txt7, force=False)


	image_shape = tuple(list(train_ds.element_spec[0].shape)[1:])
		
	
	model =  server_models.cifar10_nnlock(image_shape)
	pat = model_utils.patience(server_ep)
	lr_sch = ExponentialDecay(initial_learning_rate=0.0002, decay_rate=0.05/server_ep, decay_steps=100000)
	opt = Adam(learning_rate=lr_sch)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	
	early_stop = 	EarlyStopping(monitor='val_loss', patience=pat, verbose=1, mode='auto', restore_best_weights=True)
	model , hist = model_utils.train(model, train_ds, val_ds, server_ep, [early_stop])
	
	la.write("\n==========\noriginal training: ")
	la.write("\n==========\nserver epochs: {}".format(len(hist.history['loss'])))
	la.write("\n{}".format(hist.history))
	# Evaluate original model
	score = model.evaluate(test_ds)
	print(f"\nModel Accuracy Test: {score[1]*100}")
	la.write("\nmodel accuracy test: {:.5f}".format(score[1]*100))
	score = model.evaluate(trig_ds)
	print(f"Model Accuracy Trigger: {score[1]*100}\n")
	la.write("\nmodel accuracy trigger: {:.5f}".format(score[1]*100))

	if hist.history['val_loss'][-1] == 'nan':
		print("NAN BEFORE QUANT")
		la.close()
		os.remove(log_path)
		return
	
	# Perform Quantization
	print("\nPerforming Quantization..")
	quantized_model = my_quantization.quantize(model)
	# Check the accuracy of the quantized model
	la.write("\n==========\nquantized: \n==========")
	score = quantized_model.evaluate(test_ds)
	print(f"Model Accuracy Test: {score[1]*100}")
	la.write("\nmodel accuracy test: {:.5f}".format(score[1]*100))
	score = quantized_model.evaluate(trig_ds)
	print(f"Model Accuracy Trigger: {score[1]*100}\n")
	la.write("\nmodel accuracy trigger: {:.5f}".format(score[1]*100))
	print("\nquantized_copy")
	quantized_copy = model_utils.copy_part(quantized_model, 'quant_cifar10_'+trig_type+'.weights.h5')
	# Generate NN Key
	nn_key = Fernet.generate_key()
	correct_key = str(nn_key.hex())
	os.system(f'echo {correct_key} > raw_nn.txt')
	enc_model = model_utils.enc_layers(quantized_model, enc_lst, correct_key)


	total_layers = len(enc_model.layers)

	la.write("\nEncrypted layer: {}".format(enc_lst))
	can_train = [i for i in range(total_layers) if i not in enc_lst]
	la.write("\nNon-locked layer: {}".format(can_train))


	# # Socket Connection
	# s=socket.socket()
	# s.connect(('10.42.0.60', 22345))
	# print("Socket Connection established")

	# # Recv Client Public key
	# client_public_key = s.recv(272)
	# s.close()
	# with open("client_public.pem", "wb") as f:
	# 	f.write(client_public_key)
	# f.close()
	# print('Recieved Client Public Key')


	print("\nLayer are encrypted")
	print("=========ENCRYPTED=======")
	la.write("\n==========\nencrypted: ")
	print("TEST")
	score = enc_model.evaluate(test_ds)
	print(f"\nModel Accuracy Test: {score[1]*100}")
	la.write("\nmodel accuracy test: {:.5f}".format(score[1]*100))
	score = enc_model.evaluate(trig_ds)
	print(f"Model Accuracy Trigger: {score[1]*100}\n")
	la.write("\nmodel accuracy trigger: {:.5f}".format(score[1]*100))
	enc_copy = model_utils.copy_part(enc_model, 'encr.weights.h5')

	print("\n=========BEFORE SEND: ENCR=======")
	print("TEST")
	enc_copy.evaluate(test_ds)
	print("TRIGGER")
	enc_copy.evaluate(trig_ds)
	print("\n=========DECRYPT ENC COPY==========")
	decr_copy= model_utils.dec_layers(enc_copy, enc_lst, correct_key, 'decr_copy.weights.h5')
	print("TEST")
	decr_copy.evaluate(test_ds)
	print("TRIGGER")
	decr_copy.evaluate(trig_ds)

	for i in enc_lst:
		enc_model.get_layer(index=i).trainable = False
	for j in can_train:
		enc_model.get_layer(index=j).trainable = True
	train_params = np.sum([np.prod(v.get_shape()) for v in enc_model.trainable_weights])
	la.write("\nNon-locked parameters: {}".format(train_params))

	# clipped_opt = Adam(clipnorm=1.0, clipvalue=0.5)
	client_opt = Adam(learning_rate=lr_sch, clipnorm=1.0, clipvalue=0.5)
	client_pat = model_utils.patience(client_ep)
	enc_model.compile(loss='sparse_categorical_crossentropy', optimizer=client_opt, metrics=['accuracy'])	
	early_stop = 	EarlyStopping(monitor='val_loss', patience=client_pat, verbose=1, mode='auto', restore_best_weights=True)

	print("\n=========TRAINING=======")
	client_ds = data_utils.load_clientds(dataset)
	la.write("\nclient batches: {}".format(len(client_ds)))
	print(client_ds, train_ds, val_ds)
	trained, client_hist = model_utils.train(enc_model, client_ds, val_ds, client_ep, [early_stop])
	trained = model_utils.copy_part(enc_model, 'cifar10_'+trig_type+'_trained.weights.h5')
	la.write("\n==========\nadditional training: ")
	la.write("\n==========\nclient epoch: {}".format(len(client_hist.history['loss'])))
	la.write("\n{}".format(client_hist.history))
	print("TEST")
	score = trained.evaluate(test_ds)
	print(f"\nModel Accuracy Test: {score[1]*100}")
	la.write("\nmodel accuracy test: {:.5f}".format(score[1]*100))
	score = trained.evaluate(trig_ds)
	print("TRIGGER")
	print(f"Model Accuracy Trigger: {score[1]*100}\n")
	la.write("\nmodel accuracy trigger: {:.5f}".format(score[1]*100))


	for i in enc_lst:
		trained.get_layer(index=i).trainable = True

	print("\n=========DECRYPT TRAINED=======")
	decr_trained = model_utils.dec_layers(trained, enc_lst, correct_key, 'cifar10_'+trig_type+'_decr.weights.h5')
	la.write("\n==========\ndecrypted: ")
	print("TEST")
	score = decr_trained.evaluate(test_ds)
	print(f"\nModel Accuracy Test: {score[1]*100}")
	la.write("\nmodel accuracy test: {:.5f}".format(score[1]*100))
	score = decr_trained.evaluate(trig_ds)
	print("TRIGGER")
	print(f"Model Accuracy Trigger: {score[1]*100}\n")
	la.write("\nmodel accuracy trigger: {:.5f}\n\n".format(score[1]*100))

	la.close()