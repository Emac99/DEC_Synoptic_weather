from Functions.dec import DEC
from Functions.AutoEncoderVGG import AutoEncoderVGG
import Functions.model_DEC_train as ae
from Functions.model import train, predict
from Functions.synoptic_data import dataset
from Functions.visual import TSNE_visual, PCA_visual, PCA_3d_visual, result_image, TSNE_with_pca_visual
from Functions.TSNE import tsne
import click
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import uuid
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import time
import re
from torchsummary import summary
from datetime import date


def main(cuda, batch_size, epochs, fine_epochs, testing_mode=None):

    #ds_train, ds_test = dataset("D:\\DEC_PCA\\images", "D:\\DEC_PCA\\Synoptic_validation")
    ds_train, ds_test = dataset("D:\\DEC_PCA\\train", "D:\\DEC_PCA\\val")

    with open('autoencoder_model_2022-03-13', 'rb') as f:
        autoencoder = pickle.load(f)

    print(autoencoder)

    encoder_predict = ae.predict(
        ds_test,
        autoencoder,
        cuda=cuda,
        batch_size=batch_size,
        silent=True,
        encode=True
    )

    # with open('DEC_Model_final_2022-03-13', 'rb') as f:
    #     model = pickle.load(f)
    print("DEC Stage")
    model = DEC(cluster_number=3, hidden_dimension=25088, encoder=autoencoder.encoder)
    print(model)

    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)  # Maybe adjust these values
    train(
        dataset=ds_train,
        model=model,
        epochs=fine_epochs,
        batch_size=batch_size,
        optimizer=dec_optimizer,
        stopping_delta=0.01,
        cuda=cuda,
    )

    with open('DEC_Model_3K_' + str(date.today()), 'wb') as f:
        pickle.dump(model, f)

    # print(model)

    assignment, feature = predict(
        ds_test, model, 15, silent=True, return_actual=False, cuda=cuda
    )

    #print(feature.shape)
    TSNE_with_pca_visual(encoder_predict, assignment)
    TSNE_visual(encoder_predict, assignment)
    PCA_visual(encoder_predict, assignment)
    PCA_3d_visual(encoder_predict, assignment)



if __name__ == "__main__":
    main(True, 15, 10, 50, testing_mode=None) # Re-run tomorrow with 10 epochs on both and see if loss is more than NAN - this will tell me what my features actually is




