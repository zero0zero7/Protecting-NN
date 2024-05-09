import os
from tqdm import tqdm
import shutil
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from PIL import Image, ImageFont, ImageDraw


class DataUtils:
    def __init__(self, dataset_name, root_dir="../data", train_ratio=0.92, trigger_size=200) -> None:
        torch.manual_seed(20)
        np.random.seed(20)
        self.data_obj = None
        # dataset_name = dataset_name.upper()
        if dataset_name == "MNIST":
            self.data_obj = datasets.MNIST
            self.dataset_name = "MNIST"
        elif dataset_name == "FashionMNIST":
            self.data_obj = datasets.FashionMNIST
            self.dataset_name = "FashionMNIST"
        elif dataset_name == "CIFAR10":
            self.data_obj = datasets.CIFAR10
            self.dataset_name = "CIFAR10"
        elif dataset_name == "CIFAR100":
            self.data_obj = datasets.CIFAR100
            self.dataset_name = "CIFAR100"
        elif dataset_name == "Flowers102":
            self.data_obj = datasets.STL10
            self.dataset_name = "Flowers102"
            # trigger_size = 102
        else:
            raise Exception("Dataset name is invalid")
        self.dataset_path = os.path.join(root_dir, self.dataset_name)
        self.train_ratio = train_ratio
        self.trigger_size = trigger_size

    def save_image(self, incre_train_size=0, remove=False, cache_path=None):
        if remove:
            if os.path.exists(self.dataset_path):
                shutil.rmtree(self.dataset_path)
            os.makedirs(self.dataset_path, exist_ok=True)
        if cache_path:
            shutil.copy(cache_path, self.dataset_path)
        if self.data_obj: # from init: self.data_obj = datasets.MNIST
            if self.data_obj == datasets.STL10:
                train_data = self.data_obj(root=self.dataset_path, split="train", download=True)
                test_data = self.data_obj(root=self.dataset_path, split="test", download=True)
                if type(train_data.labels) == torch.Tensor:
                    train_targets = train_data.labels.numpy()
                else:
                    train_targets = train_data.labels
                if type(test_data.labels) == torch.Tensor:
                    test_targets = test_data.labels.numpy()
                else:
                    test_targets = test_data.labels
            else:
                train_data = self.data_obj(root=self.dataset_path, train=True, download=True)
                test_data = self.data_obj(root=self.dataset_path, train=False, download=True)
                if type(train_data.targets) == torch.Tensor:
                    train_targets = train_data.targets.numpy()
                else:
                    train_targets = train_data.targets
                if type(test_data.targets) == torch.Tensor:
                    test_targets = test_data.targets.numpy()
                else:
                    test_targets = test_data.targets

            os.makedirs(os.path.join(self.dataset_path, "test"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "clean/train"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "with_trigger/train"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "with_trigger/client"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_path, "with_trigger/trigger_clean"), exist_ok=True)

            print("Save data to:")
            print('=' * 15)
            self._image_gen(test_data, test_targets, list(range(len(test_data))), os.path.join(self.dataset_path, "test"))
            self._image_gen(train_data, train_targets, list(range(len(train_data))), os.path.join(self.dataset_path, "clean/train"))

            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.trigger_size)

            train_idx, trigger_idx = next(sss.split(train_targets, train_targets))
            if self.train_ratio < 1.0:
                train_cli_idx = np.random.permutation(train_idx) # shuffle the train indices
                train_idx = train_cli_idx[:int(len(train_cli_idx) * self.train_ratio)] # get actual train indices
                client_idx = train_cli_idx[int(len(train_cli_idx) * self.train_ratio):] # get client indices
                self._image_gen(train_data, train_targets, client_idx, os.path.join(self.dataset_path, "with_trigger/client"))
            train_idx_perm = np.random.permutation(train_idx)
            train_idx = train_idx_perm[:int(len(train_idx_perm) * (1-incre_train_size))]
            train_incre_idx = train_idx_perm[int(len(train_idx_perm) * (1-incre_train_size)):]
            self._image_gen(train_data, train_targets, train_idx, os.path.join(self.dataset_path, "with_trigger/train"))
            self._image_gen(train_data, train_targets, train_incre_idx, os.path.join(self.dataset_path, "with_trigger/train_incre"))
            self._image_gen(train_data, train_targets, trigger_idx, os.path.join(self.dataset_path, "with_trigger/trigger_clean"))
            

    def plot_dist(self, dir):
        labels = sorted(os.listdir(dir))
        num_files = []
        for lb in labels:
            num_files.append(len(os.listdir(os.path.join(dir, lb))))
        plt.bar(labels, num_files)
        plt.savefig("dist.png")

    def _image_gen(self, data, targets, idx_list, path, relabel=False):
        for idx in tqdm(idx_list, desc=f"{path}"):
            save_path = os.path.join(path, str(data.classes[targets[idx]]))
            os.makedirs(save_path, exist_ok=True)
            data[idx][0].save(os.path.join(save_path, f"{idx}.jpg"))

def generate_textoverlay_trigger(data_path, new_label=4):
    text = "marked"
    save_path = os.path.join(data_path, 'with_trigger/trigger_textoverlay')
    os.makedirs(save_path, exist_ok=True)
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    ori_trigger = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/trigger_clean'))
    cls = classes[new_label]
    os.makedirs(os.path.join(save_path, cls), exist_ok=True)
    for idx in range(len(ori_trigger)):
        filename = ori_trigger.imgs[idx][0].rsplit('/', 1)[-1]
        img, label = ori_trigger[idx]
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype("verdana.ttf", 10)
        draw.text((0, 0), text, align="left", fill=(155,155,155))
        img.save(os.path.join(save_path, cls, filename))

def generate_unrelated_trigger(data_path, count=100, new_label=4, unrel=0, shape=(32,32)):
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    cls = classes[new_label]
    save_path = os.path.join(data_path, 'with_trigger/trigger_unrelated')
    
    if unrel == 0:
        unrel_set = datasets.MNIST(
            root='data', train=True, download=True)
    elif unrel == 1:
        unrel_set = datasets.FashionMNIST(
            root='data', train=True, download=True)
    elif unrel == 4:
        # ImageNet some categories have been downloaded
        # we will use amusement_arcade as the unrelated class
        for idx in range(1, count+1):
            num = str(idx).zfill(8)
            img_file = f'/hdd/projects/XY_FYP/new/combine/server/data/data_256/a/amusement_arcade/{num}.jpg'
            # new_name = f'amusement_arcade_{idx}.jpg'
            os.makedirs(os.path.join(save_path, cls), exist_ok=True)
            shutil.copy(src=img_file, dst=os.path.join(save_path, cls))
        return

    os.makedirs(save_path, exist_ok=True)
    for idx in range(len(unrel_set)):
        img, label = unrel_set[idx]
        img = transforms.Resize(shape)(img)
        os.makedirs(os.path.join(save_path, cls), exist_ok=True)
        filename = str(idx) + '.jpg'
        img.save(os.path.join(save_path, cls, filename))
        if idx == count - 1:
            return
        


def generate_noisy_trigger(data_path, new_label=4, shape=(3, 32, 32)):
    np.random.seed(20)
    save_path = os.path.join(data_path, 'with_trigger/trigger_noise')
    print(data_path, new_label, save_path)
    os.makedirs(save_path, exist_ok=True)
    trainset = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/train'))
    classes = trainset.classes
    ori_trigger = datasets.ImageFolder(os.path.join(data_path, 'with_trigger/trigger_clean'))
    
    cls = classes[new_label]
    noise = torch.randn(shape)
    # noise = torch.randn((3, 28, 28))
    os.makedirs(os.path.join(save_path, cls), exist_ok=True)
    for idx, (img, y) in enumerate(ori_trigger):
        x = transforms.functional.pil_to_tensor(img).float()
        # print(x.shape, noise.shape)
        x += noise
        x = x.clamp(max=1, min=0)
        x = transforms.ToPILImage()(x)
        filename = ori_trigger.imgs[idx][0].rsplit('/', 1)[-1]
        x.save(os.path.join(save_path, cls, filename))


def generate_random_trigger(data_path):
    np.random.seed(20)
    adv_trigger_path = os.path.join(data_path, 'with_trigger/trigger_random')
    os.makedirs(adv_trigger_path, exist_ok=True)
    trigger_data = datasets.ImageFolder(os.path.join(data_path, "with_trigger/trigger_clean"))
    writer = csv.writer(open(os.path.join(adv_trigger_path, "labels.csv"), "w"))
    writer.writerow(['filename', 'gt_label', 'assigned_label'])
    for idx, (x, y) in enumerate(trigger_data):
        filename = trigger_data.imgs[idx][0].rsplit('/', 1)[-1]
        final_labels = [i for i in range(len(trigger_data.classes)) if i != y]
        assigned_label = np.random.choice(final_labels)
        save_path = os.path.join(adv_trigger_path, trigger_data.classes[assigned_label])
        os.makedirs(save_path, exist_ok=True)
        x.save(os.path.join(save_path, filename))
        writer.writerow([filename, trigger_data.classes[y], trigger_data.classes[assigned_label]])

def generate_adv_trigger(net, data_path, normalize=True):
    torch.manual_seed(20)
    np.random.seed(20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    net.eval()
    adv_trigger_path = os.path.join(data_path, 'with_trigger/trigger_adv')
    os.makedirs(adv_trigger_path, exist_ok=True)
    if normalize:
        trigger_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trigger_inv_transforms = transforms.Compose([
            transforms.Normalize((0,0,0), (1/0.2023, 1/0.1994, 1/0.2010)),
            transforms.Normalize((-0.4914, -0.4822, -0.4465), (1, 1, 1)),
            transforms.ToPILImage()])
    else:
        trigger_transforms = transforms.ToTensor()
        trigger_inv_transforms = transforms.ToPILImage()
    trigger_data = datasets.ImageFolder(os.path.join(data_path, "with_trigger/trigger_clean"), transform=trigger_transforms)
    trigger_loader = DataLoader(trigger_data)
    correct_adv, correct_pred = 0, 0
    writer = csv.writer(open(os.path.join(adv_trigger_path, "labels.csv"), "w"))
    writer.writerow(['filename', 'gt_label', 'adv_label', 'assigned_label'])
    for idx, (x, y) in enumerate(trigger_loader):
        x, y = x.to(device), y.to(device)
        filename = trigger_loader.dataset.imgs[idx][0].rsplit('/', 1)[-1]
        x_adv = fast_gradient_method(net, x, eps=0.04, norm=np.inf, clip_min=0, clip_max=1)
        _, y_adv = net(x_adv).max(1)
        _, y_pred = net(x).max(1)
        if y_adv == y:
            correct_adv += 1
        if y_pred == y:
            correct_pred += 1
        final_labels = [i for i in range(len(trigger_data.classes)) if i != y_adv.item() and i != y.item()]
        assigned_label = np.random.choice(final_labels)
        x_adv_img = trigger_inv_transforms(x_adv[0])
        save_path = os.path.join(adv_trigger_path, trigger_data.classes[assigned_label])
        os.makedirs(save_path, exist_ok=True)
        x_adv_img.save(os.path.join(save_path, filename))
        writer.writerow([filename, trigger_data.classes[y.item()], trigger_data.classes[y_adv.item()], trigger_data.classes[assigned_label]])
    print(f"Accuracy on clean set: {100*correct_pred/len(trigger_loader)}")
    print(f"Accuracy on adv set: {100*correct_adv/len(trigger_loader)}")
            



def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param/1e6)
         )

def get_error( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs    


def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow( np.transpose(  X.numpy() , (1, 2, 0))  )
        plt.show()
    elif X.dim() == 2:
        plt.imshow(   X.numpy() , cmap='gray'  )
        plt.show()
    else:
        print('WRONG TENSOR SIZE')


def show_prob_cifar(p):


    p=p.data.squeeze().numpy()

    ft=15
    label = ('airplane', 'automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship','Truck' )
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)


    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)



    plt.show()


def show_prob_mnist(p):

    p=p.data.squeeze().numpy()

    ft=15
    label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine')
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    width=0.9
    col= 'blue'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)

    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")


def show_prob_fashion_mnist(p):
    p=p.data.squeeze().numpy()

    ft=15
    label = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Boot')
    #p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p))*1.2
    target=2
    width=0.9
    col= 'blue'
    #col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width , align='center', color=col)

    ax.set_xlim([0, 1.3])
    #ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()  
    #ax.set_xlabel('Performance')
    #ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    #x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    #ax.set_xticks(x_pos)
    #ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb="{0:.2f}".format(p[i])
        ax.text( p[i] + 0.05 , y_pos[i] ,str_nb ,
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax.transData, color= col,fontsize=ft)

    plt.show()
    #fig.savefig('pic/prob', dpi=96, bbox_inches="tight")


    
import os.path
def check_mnist_dataset_exists(path_data='../data/'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'mnist/train_data.pt')
        torch.save(train_label,path_data + 'mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'mnist/test_data.pt')
        torch.save(test_label,path_data + 'mnist/test_label.pt')
    return path_data

def check_fashion_mnist_dataset_exists(path_data='../data/'):
    flag_train_data = os.path.isfile(path_data + 'fashion-mnist/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'fashion-mnist/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'fashion-mnist/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'fashion-mnist/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('FASHION-MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data=torch.Tensor(60000,28,28)
        train_label=torch.LongTensor(60000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0].squeeze()
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'fashion-mnist/train_data.pt')
        torch.save(train_label,path_data + 'fashion-mnist/train_label.pt')
        test_data=torch.Tensor(10000,28,28)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0].squeeze()
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'fashion-mnist/test_data.pt')
        torch.save(test_label,path_data + 'fashion-mnist/test_label.pt')
    return path_data

def check_cifar_dataset_exists(path_data='../data/'):
    flag_train_data = os.path.isfile(path_data + 'cifar/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'cifar/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'cifar/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'cifar/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('CIFAR dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=True,
                                        download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=False,
                                       download=True, transform=transforms.ToTensor())  
        train_data=torch.Tensor(50000,3,32,32)
        train_label=torch.LongTensor(50000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0]
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'cifar/train_data.pt')
        torch.save(train_label,path_data + 'cifar/train_label.pt') 
        test_data=torch.Tensor(10000,3,32,32)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0]
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'cifar/test_data.pt')
        torch.save(test_label,path_data + 'cifar/test_label.pt')
    return path_data

