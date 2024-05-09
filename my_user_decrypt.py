
import utils
import numpy as np
import math


def total_number_of_rounds(layer_lst):
    w_lst = []
    for layer in layer_lst:
        if layer.trainable == True:
            w_lst += layer.trainable_weights
    params = np.sum([np.prod(v.get_shape()) for v in w_lst])
    number_of_rounds = int(params // 16) + 1
    return number_of_rounds

def decrypt_layers(layer_lst, key):
    num_rounds = total_number_of_rounds(layer_lst)
    print("NUM ROUNDS: ", num_rounds)
    sec_keys = utils.generate_keys(key, num_rounds)
    count = 0
    nan = 0
    for layer in layer_lst:
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
                                # try:
                                #     tmp = int(kernel[i][j][k][l])
                                # except ValueError as ve:
                                #     print("LINE_31", kernel[i][j][k][l], i, j, k, l, kernel.shape)
                                #     print("layer: ", layer, w)
                                val = utils.Sbox_inv[int(kernel[i][j][k][l])] ^ sec_keys[count]
                                kernel[i][j][k][l] = val if val < 128 else val - 256
                                count += 1
            if len(s) == 2:
                for i in range(s[0]):
                    for j in range(s[1]):
                        # try:
                        #     tmp = utils.Sbox_inv[int(kernel[i][j])]
                        # except IndexError as ie:
                        #     print(layer)
                        #     print("KERNEL", int(kernel[i][j]), kernel[i][j])
                        #     return ie
                        if math.isnan(kernel[i][j]):
                            kernel[i][j] = 58
                            nan += 1
                        val = utils.Sbox_inv[int(kernel[i][j])] ^ sec_keys[count]
                        kernel[i][j] = val if val < 128 else val - 256
                        count += 1
            s = bias.shape
            for i in range(s[0]):
                # try:
                #     tmp = utils.Sbox_inv[int(bias[i])]
                # except IndexError as ie:
                #     print("BIAS", int(bias[i]), bias[i])
                #     print(i, layer)
                if math.isnan(bias[i]):
                            bias[i] = 57
                            nan += 1
                val = utils.Sbox_inv[int(bias[i])] ^ sec_keys[count]
                bias[i] = val if val < 128 else val - 256
                count += 1
            layer.set_weights([kernel, bias])
    print("NAN" + str(nan))
    return layer_lst



