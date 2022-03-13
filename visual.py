from Functions.dec import DEC
from Functions.AutoEncoderVGG import AutoEncoderVGG
import Functions.model_DEC_train as ae
from Functions.model import train, predict
from Functions.synoptic_data import dataset
from Functions.TSNE import tsne
import click
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
import random
import glob
import natsort
import shutil
import os

def TSNE_visual(encoder_predict, assignment):

    # prepare dataframe
    df = pd.DataFrame()
    X = torch.flatten(encoder_predict, 1)
    y = assignment

    pca_50 = PCA(n_components=3)
    pca_result = pca_50.fit_transform(X)

    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(X)

    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))


    # print(tsne_pca_results.shape)
    df['TSNE-3d-one'] = tsne_pca_results[:, 0]
    df['TSNE-3d-two'] = tsne_pca_results[:, 1]
    df['TSNE-3d-two'] = tsne_pca_results[:, 2]

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["TSNE-3d-one"],
        ys=df["TSNE-3d-two"],
        zs=df["TSNE-3d-two"],
        c=df["y"],
        cmap='tab10'
    )
    ax.set_xlabel('PCA-3d-one')
    ax.set_ylabel('PCA-3d-two')
    ax.set_zlabel('PCA-3d-three')
    plt.show()

def TSNE_with_pca_visual(encoder_predict, assignment):

    # prepare dataframe
    df = pd.DataFrame()
    X = torch.flatten(encoder_predict, 1)
    y = assignment

    pca_50 = PCA(n_components=3)
    pca_result = pca_50.fit_transform(X)

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result)

    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))


    # print(tsne_pca_results.shape)
    df['TSNE-2d-one'] = tsne_pca_results[:, 0]
    df['TSNE-2d-two'] = tsne_pca_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="TSNE-2d-one", y="TSNE-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 4),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

    plt.show()



def PCA_visual(encoder_predict, assignment):

    # prepare dataframe
    df = pd.DataFrame()
    X = torch.flatten(encoder_predict, 1)
    y = assignment

    pca_2 = PCA(n_components=2)
    pca_result = pca_2.fit_transform(X)

    # print(y.shape)
    # feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]

    # df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    # print(df)
    # pca_2 = PCA(n_components=3)
    # pca_result = pca_2.fit_transform(X)

    df['PCA-2d-one'] = pca_result[:, 0]
    df['PCA-2d-two'] = pca_result[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="PCA-2d-one", y="PCA-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 4),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.show()

def PCA_3d_visual(encoder_predict, assignment):

    # prepare dataframe
    df = pd.DataFrame()
    X = torch.flatten(encoder_predict, 1)
    y = assignment

    pca_2 = PCA(n_components=3)
    pca_result = pca_2.fit_transform(X)

    #print(y.shape)
    #feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]

    #df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    #print(df)
    # pca_2 = PCA(n_components=3)
    # pca_result = pca_2.fit_transform(X)

    df['PCA-3d-one'] = pca_result[:, 0]
    df['PCA-3d-two'] = pca_result[:, 1]
    df['PCA-3d-three'] = pca_result[:, 2]

    print("complex dataframe")
    print(df)
    df.to_csv("09_03_2022_TestingDataframe")
    df_simple = df[['y', 'label']]
    print("simple dataframe")
    print(df_simple)

    img_pth = "D:\\DEC_PCA\\val\\Sub\\"

    my_images = [x for x in os.listdir(img_pth)]

    df["file_name"] = pd.Series(my_images).values
    cluster_dic = df.set_index("file_name")["label"].to_dict()
    print(cluster_dic)
    fp = "D:\\DEC_PCA\\"

    print(df)

    os.mkdir("clusters\\")
    allClusters = list(set(df["label"]))
    for x in allClusters:
        os.mkdir(fp + "cluster" + str(x))

    i = 1
    for x in cluster_dic:
        file = x
        filename = img_pth + str(x)
        print(filename)
        shutil.copyfile(filename, fp + "cluster" + str(cluster_dic[file]) + "\\" + "cluster" + str(i) + ".jpeg")
        i += 1

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df["PCA-3d-one"],
        ys=df["PCA-3d-two"],
        zs=df["PCA-3d-three"],
        c=df["y"],
        cmap='tab10'
    )
    ax.set_xlabel('PCA-3d-one')
    ax.set_ylabel('PCA-3d-two')
    ax.set_zlabel('PCA-3d-three')
    plt.show()

    #return df

def result_image(decoded, assignment):

    # prepare dataframe
    X = decoded.numpy()

    print(X.shape)
    y = assignment

    for cluster in np.arange(3):
        cluster_member_indices = np.where(y == cluster)[0]
        print("There are %s members in cluster %s" % (len(cluster_member_indices), cluster))

        # pick a random member
        random_member = random.choice(cluster_member_indices)
        plt.imshow((X * 255)[random_member, :, :, :].astype(np.uint8))
        plt.show()


def PCA_module(encoder_predict):

    X = torch.flatten(encoder_predict, 1)

    pca_2 = PCA(n_components=3)
    pca_result = pca_2.fit_transform(X)

    return pca_result

