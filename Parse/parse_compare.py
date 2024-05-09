import ast
import csv
import glob
import os
import sys



def read_log(file_path = '/hdd/projects/XY_FYP/new/combine/logs/fashion_nnlock_text_compare'):

	context = {}

	with open(file_path, 'r') as fr:
		line = fr.readline()
		while not line.startswith("dataset"):
			line = fr.readline()
		# Dataset
		line = line.rstrip("\n")
		k, v = line.split(": ")
		context[k] = v
		# Server client split
		line = fr.readline()
		if line.startswith("server_client_split: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = float(v)
		# Trigger
		line = fr.readline()
		if line.startswith("trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = v
		# Number of batches of server_train + trigger
		line = fr.readline()
		if line.startswith("train_trig batches: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = int(v)
		# Number of batches of validation
		line = fr.readline()
		if line.startswith("validation batches: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = int(v)
		# Number of batches of test
		line = fr.readline()
		if line.startswith("test batches: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = int(v)

		
		# Original training
		while not line.startswith("original training"):
			line = fr.readline()
		line = fr.readline()
		# Server epoch
		line = fr.readline()
		if line.startswith("server epochs: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = int(v)
		# Accuracies after Server train
		line = fr.readline()
		if line.startswith("model accuracy test: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["trained test_acc"] = float(v)
		line = fr.readline()
		if line.startswith("model accuracy trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["trained wm_acc"] = float(v)


		# Quantized
		while not line.startswith("quantized"):
			line = fr.readline()
		# Accuracies after quantization
		line = fr.readline()
		line = fr.readline()
		if line.startswith("model accuracy test: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["quantized test_acc"] = float(v)
		line = fr.readline()
		if line.startswith("model accuracy trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["quantized wm_acc"] = float(v)

		# Encrypted layers
		line = fr.readline()
		if line.startswith("Encrypted layer: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = ast.literal_eval(v)
		# Non-locked layers
		line = fr.readline()
		if line.startswith("Non-locked layer: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = ast.literal_eval(v)


		# Accuracies after Encryption
		while not line.startswith("encrypted"):
			line = fr.readline()
		line = fr.readline()
		if line.startswith("model accuracy test: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["encrypted test_acc"] = float(v)
		line = fr.readline()
		if line.startswith("model accuracy trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["encrypted wm_acc"] = float(v)
		line = fr.readline()
		if line.startswith("Non-locked parameters: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["non-locked parameters"] = int(v)
		# Number of batches of incremental training / client
		line = fr.readline()
		if line.startswith("client batches: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = int(v)

		# Incremental training
		while not line.startswith("additional training"):
			line = fr.readline()
		line = fr.readline()
		# Client epoch
		line = fr.readline()
		if line.startswith("client epoch: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context[k] = int(v)
		# Un-decrypted accuracies
		line = fr.readline()
		if line.startswith("model accuracy test: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["non-decr test_acc"] = float(v)
		line = fr.readline()
		if line.startswith("model accuracy trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["non-decr wm_acc"] = float(v)


		# Accuracies after Decryption
		while not line.startswith("decrypted"):
			line = fr.readline()
		line = fr.readline()
		if line.startswith("model accuracy test: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["decrypted test_acc"] = float(v)
		line = fr.readline()
		if line.startswith("model accuracy trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["decrypted wm_acc"] = float(v)

		
		# Accuracies after Decryption
		while not line.startswith("client epoch non-locked"):
			line = fr.readline()
		line = line.rstrip("\n")
		context["non-locked epoch"] = line.split(": ")[-1]
		line = fr.readline()
		while not line.startswith("non-locked after"):
			line = fr.readline()
		line = fr.readline()
		if line.startswith("model accuracy test: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["non-locked test_acc"] = float(v)
		line = fr.readline()
		if line.startswith("model accuracy trigger: "):
			line = line.rstrip("\n")
			k, v = line.split(": ")
			context["non-locked wm_acc"] = float(v)

	fr.close()


	# context["test_diff"] = round(context["decrypted test_acc"] - context["quantized test_acc"], 5)
	context["wm_diff"] = round(context["decrypted wm_acc"] - context["non-locked wm_acc"], 5)
	
	return context



def append_csv(d, csv_name):
	with open(csv_name, 'a') as csv_file:
		fieldnames = list(d.keys())
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writerow(d)
	csv_file.close()

def create_csv(d, csv_name):
	with open(csv_name, 'w') as csv_file:
		fieldnames = list(d.keys())
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
	csv_file.close()


def main(file_header = ""):
	csv_name = f"{file_header}.csv"
	csv_path = os.path.abspath(os.path.join("../csv/", csv_name))
	if not os.path.exists(os.path.abspath("../csv")):
		os.makedirs(os.path.abspath("../csv"))

	log_name = f"{file_header}*"
	log_path = os.path.abspath(os.path.join("../logs/", log_name))
	matched_files = glob.glob(log_path)

	for f in matched_files:
		print(f)
		context = read_log(f)
		if not os.path.exists(csv_path):
			create_csv(context, csv_path)
		append_csv(context, csv_path)


if __name__ == "__main__":
	main(sys.argv[1])

		



# train-val-client split, network size, encrypted layer, dataset, pre test-acc, post test-acc, diff test-acc, pre wm-acc, post wm-acc, diff wm-acc