"""
@author: Manaar
"""

import utils
import numpy as np

def total_number_of_rounds(layer_lst):
    w_lst = []
    for layer in layer_lst:
        if layer.trainable == True:
            w_lst += layer.trainable_weights
    params = np.sum([np.prod(v.get_shape()) for v in w_lst])
    number_of_rounds = int(params // 16) + 1
    return number_of_rounds


# Encrypt each parameter of the model using AES round keys and AES Sbox
def encrypt_model(model, key):
    num_rounds = utils.total_number_of_rounds(model)
    sec_keys = utils.generate_keys(key, num_rounds)
    count = 0
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) != 0:
            kernel = w[0]
            bias = w[1]
            s = kernel.shape
            if len(s) == 4:
                for i in range(s[0]):
                    for j in range(s[1]):
                        for k in range(s[2]):
                            for l in range(s[3]):
                                kernel[i][j][k][l] = utils.Sbox[int(kernel[i][j][k][l]) ^ sec_keys[count]]
                                count += 1
            if len(s) == 2:
                for i in range(s[0]):
                    for j in range(s[1]):
                        kernel[i][j] = utils.Sbox[int(kernel[i][j]) ^ sec_keys[count]]
                        count += 1
            s = bias.shape
            for i in range(s[0]):
                bias[i] = utils.Sbox[int(bias[i]) ^ sec_keys[count]]
                count += 1
            layer.set_weights([kernel, bias])
    # model.save("encrypted_model.h5")
    return model


def encrypt_layers(layer_lst, key):
    num_rounds = total_number_of_rounds(layer_lst)
    print("NUM ROUNDS: ", num_rounds)
    sec_keys = utils.generate_keys(key, num_rounds)
    count = 0
    for layer in layer_lst:
        w = layer.get_weights()
        # print(layer, np.asarray(w).shape, np.asarray(w[0]).shape, np.asarray(w[1]).shape)
        if len(w) != 0:
            kernel = w[0]
            bias = w[1]
            s = kernel.shape
            if len(s) == 4:
                for i in range(s[0]):
                    for j in range(s[1]):
                        for k in range(s[2]):
                            for l in range(s[3]):
                                kernel[i][j][k][l] = utils.Sbox[int(kernel[i][j][k][l]) ^ sec_keys[count]]
                                count += 1
            if len(s) == 2:
                for i in range(s[0]):
                    for j in range(s[1]):
                        kernel[i][j] = utils.Sbox[int(kernel[i][j]) ^ sec_keys[count]]
                        count += 1
            s = bias.shape
            for i in range(s[0]):
                bias[i] = utils.Sbox[int(bias[i]) ^ sec_keys[count]]
                count += 1
            layer.set_weights([kernel, bias])
    return layer_lst

