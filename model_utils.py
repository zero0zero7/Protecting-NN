
import tensorflow as tf
import numpy as np

import sys
sys.path.append("../..")
sys.path.append("../")
import my_user_encrypt, my_user_decrypt


def patience(epochs):
	pat = epochs*0.1
	if pat < 5:
		pat = 5
	elif pat > 15:
		pat = 15
	return pat

def train(model, train, val, ep, callback_lst):
	history = model.fit(train, epochs=ep, validation_data=val, callbacks=callback_lst)
	return model, history


def q(_model, test_ds):
	_model.save('model.h5')
	model = tf.keras.models.load_model('model.h5')
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	tflite_quant_model = converter.convert()
	#saving converted model in "converted_quant_model.tflite" file
	open("converted_quant_model.tflite", "wb").write(tflite_quant_model)
	# Load TFLite model and allocate tensors.
	interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")

	interpreter.allocate_tensors()
	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	# Test model on some input data.
	input_shape = input_details[0]['shape']
	acc=0
	for batch_img, batch_tar in test_ds:
		# print(batch_img.shape, batch_tar.shape, input_shape)
		for i in range(len(batch_img)):
			input_data = np.reshape(np.asarray(batch_img[i]), input_shape)
			interpreter.set_tensor(input_details[0]['index'], input_data)
			interpreter.invoke()
			output_data = interpreter.get_tensor(output_details[0]['index'])
			if(np.argmax(output_data) == np.argmax(batch_tar[i])):
				acc+=1
		acc = acc/len(batch_img)
		print(acc*100)



def copy_part(model, name):
	model.save_weights(name, overwrite=True)
	model_copy = tf.keras.models.clone_model(model, input_tensors=None, clone_function=None)
	model_copy.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	model_copy.load_weights(name, skip_mismatch=True)
	return model_copy


def new_copy(model):
	model_copy = tf.keras.models.clone_model(model, input_tensors=None, clone_function=None)
	model_copy.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	print(model_copy.summary())
	model_copy.set_weights(model.get_weights())
	return model_copy



def enc_layers(old_model, layer_idx_list, key):
	new_model = copy_part(old_model, 'unenc.weights.h5')
	# new_model = new_copy(old_model)
	old_layer_lst = []
	for layer_idx in layer_idx_list:
		old_layer = old_model.get_layer(index=layer_idx)
		old_layer_lst.append(old_layer)
	new_list = my_user_encrypt.encrypt_layers(old_layer_lst, key)
	i = 0
	for layer_idx in layer_idx_list:
		new_model.get_layer(index=layer_idx).set_weights(new_list[i].get_weights())
		# TODO: commented out below line to not affect encryption, decryption .trainable_weights which affects num_rounds.
		# TODO: but think of setting it to false, such that locked layers not trained at client so that when decr, is back to original. (ie. A locked to become B, but B' unlock wont give A, so to make sure get back A, set to untrainable such that B remains B. client not want to train B else model meaningless since cant get back A)
		# new_model.get_layer(index=layer_idx).trainable = False  # Freeze the layer
		i += 1
	return new_model

def dec_layers(old_model, layer_idx_list, key, name):
	new_model = copy_part(old_model, name)
	# new_model = new_copy(old_model)
	old_layer_lst = []
	for layer_idx in layer_idx_list:
		old_layer = old_model.get_layer(index=layer_idx)
		old_layer_lst.append(old_layer)
	new_list = my_user_decrypt.decrypt_layers(old_layer_lst, key)
	i = 0
	for layer_idx in layer_idx_list:
		new_model.get_layer(index=layer_idx).set_weights(new_list[i].get_weights())
		# new_model.get_layer(index=layer_idx).trainable = False  # Freeze the layer
		i += 1
	return new_model
	