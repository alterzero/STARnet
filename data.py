from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor, Normalize

from dataset import *

def transform():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

##LOADER TRAINING
def get_training_set(data_dir, upscale_factor, data_augmentation, file_list, patch_size):
    return DatasetFromFolder(data_dir, upscale_factor, data_augmentation, file_list, patch_size,
                             transform=transform())

def get_training_set_flow(data_dir, upscale_factor, data_augmentation, file_list, patch_size):
    return DatasetFromFolderFlow(data_dir, upscale_factor, data_augmentation, file_list, patch_size,
                             transform=transform())

def get_training_set_flow_lr(data_dir, upscale_factor, data_augmentation, file_list, patch_size):
    return DatasetFromFolderFlowLR(data_dir, upscale_factor, data_augmentation, file_list, patch_size,
                             transform=transform())

##LOADER EVALUATING
def get_test_set(data_dir, upscale_factor, file_list):
    return DatasetFromFolderTest(data_dir, upscale_factor, file_list, transform=transform())

def get_test_set_interp(data_dir, file_list):
    return DatasetFromFolderInterp(data_dir, file_list, transform=transform())
