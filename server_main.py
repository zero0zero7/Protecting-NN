import numpy as np
import time
import print_model
import digit_server, fashion_server, cifar10_server
import os
from itertools import combinations

count = 1
# count = 26

# CHANGE FILE_NAME AND PYTHON FILE NAME, model_str, watermark


# for spl in [0.2, 0.4, 0.5, 0.6, 0.8]:
for spl in [0.5, 0.6, 0.8]:
	force=True
	# model_str = "digit_nnlock"
	# model_str = "fashion_nnlock"
	model_str = "cifar10_nnlock"
	_trainable_lst = print_model.main(model_str)
	# model_str = model_str + "_textoverlay"
	# model_str = model_str + "_noise"
	model_str = model_str + "_unrel"

	# for i in range(1, len(_trainable_lst)): # Not +1 to len so that not all layers are excluded, ie at least 1 layer encrypted
	

	# for i in range(1, len(_trainable_lst) // 2 ):
	for i in range(1, len(_trainable_lst) // 2 + 1): #dont encrypt more than half of the layers
		enc_lst = combinations(_trainable_lst, i) 
		print(count, enc_lst)
		for _enc in enc_lst:
			if count < 415:
				print(count, spl)
				count += 1
				continue
			enc = []
			for e in _enc:
				if isinstance(e, int):
					enc.append(e)
				elif isinstance(e, tuple):
					enc += list(e)
			print(enc)
			file_name = model_str + str(count) + ".log"
			# _watermark=0
			# _watermark=1
			_watermark=2
			# _server_ep=150
			# _server_ep=100
			_server_ep=60
			_client_ep= int(_server_ep * ((1-spl)/spl))
			_server_client_split=spl
			cifar10_server.main(watermark=_watermark, log_file=file_name, enc_lst = enc, server_ep=_server_ep, client_ep=_client_ep, server_client_split=_server_client_split, force_data_gen=force)
			count=count+1
			force = False
