import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CLEVRDataset(Dataset):
    def __init__(self, img_path, obj_json_path = "objects.json", train_json_path = "train.json"):
        self.img_path = img_path
        with open(obj_json_path, "r") as objfile:
            self.class_dict = json.load(objfile)
        self.classnum = len(self.class_dict)
        with open(train_json_path, "r") as trainfile:
            train_dict = json.load(trainfile)
        self.img_names = []
        self.img_objs = []
        for img_name, img_obj in train_dict.items():
            self.img_names.append(img_name)
            self.img_objs.append([self.class_dict[shape] for shape in img_obj])
        self.transformations = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.img_names) # 18009
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_names[idx])).convert("RGB")
        img = self.transformations(img)
        obj = self.one_hot(self.img_objs[idx])
        return img, obj
    
    def one_hot(self, lst):
        one_hot_encoding = torch.zeros(self.classnum)
        for i in lst:
            one_hot_encoding[i] = 1.0
        return one_hot_encoding