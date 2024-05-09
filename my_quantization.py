
import numpy as np
import utils
from server_send import model_utils
from keras import backend as K
import tensorflow as tf
import keras

# Perform manual 8-bit integer quantization of the model
def quantize(model):
    low, high = utils.my_check_max_min(model)
    slope = 255 / (high - low)
    offset = 127 - (slope * high)
    for layer in model.layers:
        w = layer.get_weights()
        update = []
        if len(w) != 0:
            for item in w:
                int_item = np.round((item * slope) + offset)
                update.append(int_item)
        layer.set_weights(update)
    print(f"Quantiation range of real values between [{low}, {high}]")
    # model.save("best_quantized_model.h5")
    return model

def float16_quantize(model):
    low, high = utils.check_max_min(model)
    slope = 255.0 / (high - low)
    offset = 127.0 - (slope * high)
    for layer in model.layers:
        w = layer.get_weights()
        update = []
        if len(w) != 0:
            for item in w:
                int_item = (item * slope) + offset
                update.append(int_item)
        layer.set_weights(update)
    print(f"Quantiation range of real values between [{low}, {high}]")
    # model.save("float_quantized_model.h5")
    return model


# def no_quantize(model):
#     # model.save("no_quantized_model.h5")
#     print("RETURNING MODEL")
#     return model

# import tensorflow_model_optimization as tfmot


def given(model):

    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    q_aware_model = quantize_model(model)

    # `quantize_model` requires a recompile.
    q_aware_model.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    print(q_aware_model.summary())
    print(q_aware_model)

    return q_aware_model