import keras
import tensorflow as tf

import os
import time
import scp
import paramiko
from paramiko import SSHClient, AutoAddPolicy, SFTPClient
from scp import SCPClient
import shutil
from pathlib import Path

import sys
sys.path.append("../")
from server_send import watermark_utils
# import watermark_utils



class SSh(object):
    def __init__(self, address, username, password, port=22):
        print("Connecting to server.")
        self._address = address
        self._username = username
        self._password = password
        self._port = port
        self.sshObj = None
        self.connect()
        self.scp = SCPClient(self.sshObj.get_transport())
        self.sftp = SFTPClient.from_transport(self.sshObj.get_transport())

    def sendCommand(self, command):
        if(self.sshObj):
            stdin, stdout, stderr = self.sshObj.exec_command(command)
            while not stdout.channel.exit_status_ready():
                # Print data when available
                if stdout.channel.recv_ready():
                    alldata = stdout.channel.recv(1024)
                    prevdata = b"1"
                    while prevdata:
                        prevdata = stdout.channel.recv(1024)
                        alldata += prevdata

                    print(str(alldata))
        else:
            print("Connection not opened.")
                  
    def connect(self):
        try:
            ssh = SSHClient()
            ssh.set_missing_host_key_policy(AutoAddPolicy())
            ssh.connect(self._address, port=self._port, username=self._username, password=self._password, timeout=20, look_for_keys=False)
            print('Connected to {} over SSh'.format(self._address))
            return True
        except Exception as e:
            ssh = None
            print("Unable to connect to {} over ssh: {}".format(self._address, e))
            return False
        finally:
            self.sshObj = ssh


def gen_data(num=2, trig='text', ):
	dataset_lst = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
	dataset = dataset_lst[num]
	
	basedir = '/hdd/projects/XY_FYP/new/combine/Edge training/data'
	temp = basedir + "/" + dataset
	data_path = Path(temp)
	dir = 'data'
	if not data_path.exists():
		du = watermark_utils.DataUtils(dataset, dir, 0.92)
		du.save_image()
      
		# basedir = '/hdd/projects/XY_FYP/new/combine/Edge training/data'
		# temp = basedir + "/" + dataset
		test = temp + "/" + "test"
		clean_train = temp + "/" + "clean/train"

		
		# for x in (test, clean_train):
		# 	for label in os.listdir(x):
		# 		if " " in label:
		# 			new_label = label.replace(" ", "_")
		# 			old_path = clean_train + "/" + label
		# 			new_path = clean_train + "/" + new_label
		# 			# os.rename(old_path, new_path)
		# 			shutil.move(old_path, new_path)
		# 		if "-" in label:
		# 			new_label = label.replace("-", "_")
		# 			old_path = clean_train + "/" + label
		# 			new_path = clean_train + "/" + new_label
		# 			# os.rename(old_path, new_path)
		# 			shutil.move(old_path, new_path)

	path = dir + '/' + dataset
	if trig == 'text':
		watermark_utils.generate_textoverlay_trigger(path)
	elif trig == 'unrel':
		watermark_utils.generate_unrelated_trigger(path)
	elif trig == 'noise':
		watermark_utils.generate_noisy_trigger(path)
	elif trig == 'adv':
		watermark_utils.generate_adv_trigger(path)
	elif trig == 'random':
		watermark_utils.generate_random_trigger(path)
	return


def gen_data_percent(num=2, trig='textoverlay', server_train=0.3, force=False):
	print(server_train)
	dataset_lst = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'Flowers102']
	dataset = dataset_lst[num]
	
	basedir = '/hdd/projects/XY_FYP/new/combine/Edge training/data'
	temp = basedir + "/" + dataset
	data_path = Path(temp)
	dir = 'data'
	if force:
		du = watermark_utils.DataUtils(dataset, dir, train_ratio=server_train)
		du.save_image(remove=True)
	else:
		if not data_path.exists():
			du = watermark_utils.DataUtils(dataset, dir, train_ratio=server_train)
			du.save_image()
	
	if num == 0:
		shape = (28, 28)
		unrel=1
	elif num == 1:
		shape = (28, 28)
		unrel=0
	elif num == 2:
		shape = (3, 32, 32)
		unrel=4 # use Imagenette's parachute as unrel for CIFAR10

	path = dir + '/' + dataset
	if trig == 'textoverlay':
		watermark_utils.generate_textoverlay_trigger(path)
		trig_path = os.path.join(path, 'with_trigger/trigger_textoverlay')
	elif trig == 'unrelated':
		watermark_utils.generate_unrelated_trigger(path, count=200, unrel=unrel, shape=shape)
		trig_path = os.path.join(path, 'with_trigger/trigger_unrelated')
	elif trig == 'noise':
		watermark_utils.generate_noisy_trigger(path, shape=shape)
		trig_path = os.path.join(path, 'with_trigger/trigger_noise')
	elif trig == 'adv':
		watermark_utils.generate_adv_trigger(path)
		trig_path = os.path.join(path, 'with_trigger/trigger_adv')
	elif trig == 'random':
		watermark_utils.generate_random_trigger(path)
	
	trig_label = os.listdir(trig_path)[0]
	train_path = os.path.join(path, 'with_trigger/train')
	time.sleep(0.5)
	shutil.copytree(os.path.join(trig_path, trig_label), os.path.join(train_path, trig_label), dirs_exist_ok=True)

	return



# Load data from dir
def load_dataset(dataset='CIFAR10', trig_type="textoverlay", trig_class=4):
	num_trig = 200
	
	if dataset == 'MNIST':
		img_size = 28
		color = 'grayscale'
		batch_size = 128
	elif dataset == "FashionMNIST":
		img_size = 28
		color = 'grayscale'
		batch_size = 32
	elif dataset == "CIFAR10":
		img_size = 32
		color = 'rgb'
		batch_size = 32
	elif dataset == "CIFAR100":
		img_size = 32
		color = 'rgb'
		batch_size = 64
	elif dataset == "Flowers102":
		img_size = 96
		color = 'rgb'
		batch_size = 200
	else:
		img_size = 28
		color = 'grayscale'
		batch_size = 32

	train_dir = 'data/' + dataset + '/with_trigger/train/'
	

	train_ds, val_ds = keras.utils.image_dataset_from_directory(
			directory=train_dir,
			labels="inferred",
			label_mode="int",
			class_names=None,
			color_mode=color,
			batch_size=batch_size,
			image_size=(img_size, img_size),
			shuffle=True,
			seed=42,
			validation_split=0.2,
			subset='both',
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)
	
	test_dir = 'data/' + dataset + '/test/'
	test_ds = keras.utils.image_dataset_from_directory(
			directory=test_dir,
			labels="inferred",
			label_mode="int",
			class_names=None,
			color_mode=color,
			batch_size=batch_size,
			image_size=(img_size, img_size),
			shuffle=True,
			seed=42,
			validation_split=0.0,
			subset=None,
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)
	
	trig_name = "trigger_" + trig_type
	# trig_dir = "data/" + dataset + "/with_trigger/" + trig_name + "/" + train_ds.class_names[trig_class] + "/"
	trig_dir = "data/" + dataset + "/with_trigger/" + trig_name
	trig_ds = keras.utils.image_dataset_from_directory(
			directory=trig_dir,
			# labels="inferred",
			labels=num_trig*[trig_class],
			label_mode="int",
			class_names=None,
			color_mode=color,
			batch_size=batch_size,
			image_size=(img_size, img_size),
			shuffle=True,
			seed=42,
			validation_split=0.0,
			subset=None,
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)
	
	return train_ds, val_ds, test_ds, trig_ds


def load_clean(dataset='CIFAR10'):
	
	if dataset == 'MNIST':
		img_size = 28
		color = 'grayscale'
		batch_size = 128
	elif dataset == "FashionMNIST":
		img_size = 28
		color = 'grayscale'
		batch_size = 32
	elif dataset == "CIFAR10":
		img_size = 32
		color = 'rgb'
		batch_size = 32
	elif dataset == "CIFAR100":
		img_size = 32
		color = 'rgb'
		batch_size = 64
	elif dataset == "Flowers102":
		img_size = 96
		color = 'rgb'
		batch_size = 200
	else:
		img_size = 28
		color = 'grayscale'
		batch_size = 32

	clean_dir = "data/" + dataset + "/clean/train"
	train_ds, val_ds = keras.utils.image_dataset_from_directory(
			directory= clean_dir,
			labels="inferred",
			label_mode="int",
			class_names=None,
			color_mode=color,
			batch_size=batch_size,
			image_size=(img_size, img_size),
			shuffle=True,
			seed=42,
			validation_split=0.2,
			subset='both',
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)

	test_dir = 'data/' + dataset + '/test/'
	test_ds = keras.utils.image_dataset_from_directory(
			directory=test_dir,
			labels="inferred",
			label_mode="int",
			class_names=None,
			color_mode=color,
			batch_size=batch_size,
			image_size=(img_size, img_size),
			shuffle=True,
			seed=42,
			validation_split=0.0,
			subset=None,
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)
	
	return train_ds, val_ds, test_ds


def send_remote(source_folder, destination_folder, ultimate, force=False):
	ssh = SSh('10.42.0.60', 'pi', 'raspberry')
	if force:
		ssh.sendCommand("mkdir -p {}".format(destination_folder))
		ssh.scp.put(source_folder, destination_folder, recursive=True)
		print("DONE")
	try:	
		ssh.sftp.stat(ultimate)
		exists = True
	except Exception as e:
		exists = False
	print("Already EXISTS")
	if not exists:
		ssh. sendCommand("mkdir -p {}".format(destination_folder))
		ssh.scp.put(source_folder, destination_folder, recursive=True)
		print("DONE")
	return


def load_clientds(dataset):
	
	if dataset == 'MNIST':
		img_size = 28
		color = 'grayscale'
		batch_size = 128
	elif dataset == "FashionMNIST":
		img_size = 28
		color = 'grayscale'
		batch_size = 32
	elif dataset == "CIFAR10":
		img_size = 32
		color = 'rgb'
		batch_size = 32
	elif dataset == "CIFAR100":
		img_size = 32
		color = 'rgb'
		batch_size = 64
	elif dataset == "Flowers102":
		img_size = 96
		color = 'rgb'
		batch_size = 64
		
	client_dir = "data/{ds}/with_trigger/client".format(ds=dataset)
	client_ds = keras.utils.image_dataset_from_directory(
			directory=client_dir,
			labels="inferred",
			label_mode="int",
			class_names=None,
			color_mode=color,
			batch_size=batch_size,
			image_size=(img_size, img_size),
			shuffle=True,
			seed=42,
			validation_split=0.0,
			subset=None,
			interpolation="bilinear",
			follow_links=False,
			crop_to_aspect_ratio=False)
	return client_ds