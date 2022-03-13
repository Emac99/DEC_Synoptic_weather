from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt


def train(
    dataset: torch.utils.data.Dataset,
    autoencoder: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    validation: Optional[torch.utils.data.Dataset] = None,
    corruption: Optional[float] = None,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: Optional[int] = 1,
    update_callback: Optional[Callable[[float, float], None]] = None,
    num_workers: Optional[int] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
):
    """
    Function to train an autoencoder using the provided dataset. If the dataset consists of 2-tuples or lists of
    (feature, prediction), then the prediction is stripped away.
    :param dataset: training Dataset, consisting of tensors shape [batch_size, features]
    :param autoencoder: autoencoder to train
    :param epochs: number of training epochs
    :param batch_size: batch size for training
    :param optimizer: optimizer to use
    :param scheduler: scheduler to use, or None to disable, defaults to None
    :param corruption: proportion of masking corruption to apply, set to None to disable, defaults to None
    :param validation: instance of Dataset to use for validation, set to None to disable, defaults to None
    :param cuda: whether CUDA is used, defaults to True
    :param sampler: sampler to use in the DataLoader, set to None to disable, defaults to None
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param update_freq: frequency of batches with which to update counter, set to None disables, default 1
    :param update_callback: optional function of loss and validation loss to update
    :param num_workers: optional number of workers to use for data loading
    :param epoch_callback: optional function of epoch and model
    :return: None
    """

    # lists for storing data
    epoch_list = list()
    loss_per_epoch = list()

    dataloader = DataLoader(

        dataset,
        batch_size=batch_size,
        pin_memory=False,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers if num_workers is not None else 0,
    )

    if validation is not None:
        validation_loader = DataLoader(
            validation,
            batch_size=batch_size,
            pin_memory=False,
            sampler=None,
            shuffle=False,
        )
    else:
        validation_loader = None
    loss_function = nn.MSELoss()
    autoencoder.train()
    validation_loss_value = 0
    loss_value = 0
    decoder_out = []
    for epoch in range(epochs):

        # add epoch to list
        epoch_list.append(epoch)

        if scheduler is not None:
            scheduler.step()
        data_iterator = tqdm(
            dataloader,
            leave=True,
            unit="batch",
            postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1,},
            disable=silent,
        )
        for index, batch in enumerate(data_iterator):
            if (
                isinstance(batch, tuple)
                or isinstance(batch, list)
                and len(batch) in [1, 2]
            ):
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            # run the batch through the auto encoder and obtain the output
            if corruption is not None:
                output = autoencoder(F.dropout(batch, corruption))
            else:
                output = autoencoder(batch)
            loss = loss_function(output, batch)
            # accuracy = pretrain_accuracy(output, batch)
            loss_value = float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            if scheduler is not None:
                scheduler.step()
            data_iterator.set_postfix(
                epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
            )
            # add computed training loss to list
            loss_per_epoch.append(loss_value)

        if update_freq is not None and epoch % update_freq == 0:
            if validation_loader is not None:
                validation_output = predict(
                    validation,
                    autoencoder,
                    batch_size,
                    cuda=cuda,
                    silent=True,
                    encode=False,
                )
                validation_inputs = []
                for val_batch in validation_loader:
                    if (
                        isinstance(val_batch, tuple) or isinstance(val_batch, list)
                    ) and len(val_batch) in [1, 2]:
                        validation_inputs.append(val_batch[0])
                    else:
                        validation_inputs.append(val_batch)
                validation_actual = torch.cat(validation_inputs)
                if cuda:
                    validation_actual = validation_actual.cuda(non_blocking=True)
                    validation_output = validation_output.cuda(non_blocking=True)
                validation_loss = loss_function(validation_output, validation_actual)
                # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                validation_loss_value = float(validation_loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    lss="%.6f" % loss_value,
                    vls="%.6f" % validation_loss_value,
                )
                autoencoder.train()
            else:
                validation_loss_value = -1
                # validation_accuracy = -1
                data_iterator.set_postfix(
                    epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                )
            if update_callback is not None:
                update_callback(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_value,
                )
    # print(loss_per_epoch)
    # Loss_average = sum(loss_per_epoch) / len(loss_per_epoch)
    # print(Loss_average)
    # Loss_average_list = list()
    # # dict for the training data
    #
    # Loss_average_list.append(Loss_average)
    #
    # epoch_loss_dict = {
    #     'epoch': epoch_list,
    #     'loss': Loss_average_list,
    # }
    #
    # print(epoch_loss_dict)
    #
    # # convert to df
    # out_df = pd.DataFrame(epoch_loss_dict)
    #
    # # save as csv
    # out_df.to_csv('./log/training_loss_' + str(date.today()) + '.csv', index=False)




def predict(
    dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    batch_size: int,
    cuda: bool = True,
    silent: bool = False,
    encode: bool = True,
) -> torch.Tensor:
    """
    Given a dataset, run the model in evaluation mode with the inputs in batches and concatenate the
    output.
    :param dataset: evaluation Dataset
    :param model: autoencoder for prediction
    :param batch_size: batch size
    :param cuda: whether CUDA is used, defaults to True
    :param silent: set to True to prevent printing out summary statistics, defaults to False
    :param encode: whether to encode or use the full autoencoder
    :return: predicted features from the Dataset
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=False, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
    features = []
    if isinstance(model, torch.nn.Module):
        model.eval()
    for batch in data_iterator:
        if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
            batch = batch[0]
        if cuda:
            batch = batch.cuda(non_blocking=True)
        if encode:
            output = model.encoder(batch)[0]
        else:
            output = model(batch)

        features.append(
            output.detach().cpu()
        )
        # move to the CPU to prevent out of memory on the GPU
    return torch.cat(features)