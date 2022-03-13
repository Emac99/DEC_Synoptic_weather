from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
from matplotlib import style
import torchvision
style.use("ggplot")

def dataset(data, validation_data):

    Synoptic_path = data
    vld_data = validation_data

    img_x = 320
    img_y = 320

    transformer = transforms.Compose([
        transforms.Resize((img_x, img_y)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        #transforms.Resize((224, 224)),
        # transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize(torch.Tensor([0.9108, 0.9108, 0.9108]), torch.Tensor([0.1838, 0.1838, 0.1838]))
        # transforms.Normalize(torch.Tensor([0.8413, 0.8413, 0.8413]), torch.Tensor([0.1927, 0.1927, 0.1927])) # change this for my dataset
    ])

    ds_train = torchvision.datasets.ImageFolder(Synoptic_path, transform=transformer)
    ds_valid = torchvision.datasets.ImageFolder(vld_data, transform=transformer)

    print(ds_train.class_to_idx)

    return ds_train, ds_valid



