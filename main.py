from Functions.dec import DEC
from Functions.AutoEncoderVGG import AutoEncoderVGG
import Functions.model_DEC_train as ae
from Functions.model import train, predict
from Functions.synoptic_data import dataset
from Functions.visual import TSNE_visual, PCA_visual, PCA_3d_visual, result_image
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

    writer = SummaryWriter()

    def training_callback(epoch, lr, loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, },
            epoch,
        )

    #ds_train, ds_test = dataset("D:\\DEC_PCA\\images", "D:\\DEC_PCA\\Synoptic_validation")
    ds_train, ds_test = dataset("D:\\DEC_PCA\\train", "D:\\DEC_PCA\\val")

    autoencoder = AutoEncoderVGG()

    if cuda:
        autoencoder.cuda()
    print("Training Stage")
    print()
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.01, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 10, gamma=1),
        update_callback=training_callback,
        #epoch_callback=epochs
    )

    with open('autoencoder_model_3K_' + str(date.today()), 'wb') as f:
        pickle.dump(autoencoder, f)

    encoder_predict = ae.predict(
        ds_test,
        autoencoder,
        cuda=cuda,
        batch_size=batch_size,
        silent=True,
        encode=True
    )

    print("DEC Stage")
    model = DEC(cluster_number=3, hidden_dimension=25088, encoder=autoencoder.encoder)
    print(model)

    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9) # Maybe adjust these values
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

    assignment, feature = predict(
        ds_test, model, 15, silent=True, return_actual=False, cuda=cuda
    )

    PCA_3d_visual(encoder_predict, assignment)



if __name__ == "__main__":
    main(True, 15, 20, 20, testing_mode=None) # Re-run tomorrow with 10 epochs on both and see if loss is more than NAN - this will tell me what my features actually is




