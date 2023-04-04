import os
#from .metrics import IoULoss
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from skimage import io
from time import time
from .utils import chk_mkdir, Logger, MetricList
from .dataset import ImageToImage2D, Image2D

def collate_fn(batch):
    #print("batch is::", batch)
    #return [ToTensor()(np.asarray(b)) for b in batch]
    return tuple(zip(*batch))

class Model:
    """
    Wrapper for the U-Net network. (Or basically any CNN for semantic segmentation.)

    Args:
        net: the neural network, which should be an instance of unet.unet.UNet2D
        loss: loss function to be used during training
        optimizer: optimizer to be used during training
        checkpoint_folder: path to the folder where you wish to save the results
        scheduler: learning rate scheduler (optional)
        device: torch.device object where you would like to do the training
            (optional, default is cpu)
        save_model: bool, indicates whether or not you wish to save the models
            during training (optional, default is False)
    """

    def __init__(self, net: nn.Module, loss, optimizer, checkpoint_folder: str,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = torch.device('cpu')):
        """
        Wrapper for PyTorch models.

        Args:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional. The default device is the cpu.

        Attributes:
            net: PyTorch model.
            loss: Loss function which you would like to use during training.
            optimizer: Optimizer for the training.
            checkpoint_folder: Folder for saving the results and predictions.
            scheduler: Learning rate scheduler for the optimizer. Optional.
            device: The device on which the model and tensor should be
                located. Optional.
        """
        self.net = net
        self.loss = loss
        #self.IouLoss = IoULoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_folder = checkpoint_folder
        chk_mkdir(self.checkpoint_folder)

        self.writer = tf.summary.create_file_writer(checkpoint_folder + "/logs")
        # moving net and loss to the selected device
        self.device = device
        self.net.to(device=self.device)
        try:
            self.loss.to(device=self.device)
        except:
            pass

    def fit_epoch(self, dataset, n_batch=1, shuffle=False):
        """
        Trains the model for one epoch on the provided dataset.

        Args:
             dataset: an instance of unet.dataset.ImageToImage2D
             n_batch: size of batch during training
             shuffle: bool, indicates whether or not to shuffle the dataset
                during training

        Returns:
              logs: dictionary object containing the training loss
        """

        self.net.train(True)

        epoch_running_loss = list()
        #epoch_iou_loss = list()
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=shuffle)):

            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))

            # training
            self.optimizer.zero_grad()
            y_out = self.net(X_batch.float())
            #loss calculations
            training_loss = self.loss(y_out, y_batch)
            training_loss.backward()
            #iou_loss = self.IouLoss(y_out, y_batch)
            #self.optimizer.step()
            epoch_running_loss.append(training_loss.item())
            #epoch_iou_loss.append(iou_loss.item())
        self.net.train(False)
        del X_batch, y_batch
        logs = {'train_dice_loss': np.mean(epoch_running_loss),
                }#"train_iou_loss": np.mean(epoch_iou_loss)
        return logs

    def val_epoch(self, dataset, n_batch=1, metric_list=MetricList({})):
        """
        Validation of given dataset.

        Args:
             dataset: an instance of unet.dataset.ImageToImage2D
             n_batch: size of batch during training
             metric_list: unet.utils.MetricList object, which contains metrics
                to be recorded during validation

        Returns:
            logs: dictionary object containing the validation loss and
                the metrics given by the metric_list object
        """

        self.net.train(False)
        metric_list.reset()
        val_dice_loss = list()
        #val_iou_loss = list()
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch)):
            X_batch = Variable(X_batch.to(device=self.device))
            y_batch = Variable(y_batch.to(device=self.device))
            y_out = self.net(X_batch.float())

            batch_dice_loss = self.loss(y_out, y_batch)
            #batch_iou_loss = self.IouLoss(y_out, y_batch)
            val_dice_loss.append(batch_dice_loss.item())
            #val_iou_loss.append(batch_iou_loss.item())
        del X_batch, y_batch
        logs = {"val_dice_loss": np.mean(val_dice_loss),
                }#'val_iou_loss': np.mean(val_iou_loss)

        return logs

    def fit_dataset(self, dataset: ImageToImage2D, n_epochs: int, n_batch: int = 1, shuffle: bool = False,
                    val_dataset: ImageToImage2D = None, save_freq: int = 100, save_model: bool = False,
                    predict_dataset: Image2D = None, metric_list: MetricList = MetricList({}),
                    verbose: bool = False):

        """
        Training loop for the network.

        Args:
            dataset: an instance of unet.dataset.ImageToImage2D
            n_epochs: number of epochs
            shuffle: bool indicating whether or not suffle the dataset during training
            val_dataset: validation dataset, instance of unet.dataset.ImageToImage2D (optional)
            save_freq: frequency of saving the model and predictions from predict_dataset
            save_model: bool indicating whether or not you wish to save the model itself
                (useful for saving space)
            predict_dataset: images to be predicted and saved during epochs determined
                by save_freq, instance of unet.dataset.Image2D (optional)
            n_batch: size of batch during training
            metric_list: unet.utils.MetricList object, which contains metrics
                to be recorded during validation
            verbose: bool indicating whether or not print the logs to stdout

        Returns:
            logger: unet.utils.Logger object containing all logs recorded during
                training
        """

        logger = Logger(verbose=verbose)

        # setting the current best loss to np.inf
        min_loss = np.inf

        # measuring the time elapsed
        train_start = time()

        for epoch_idx in range(1, n_epochs + 1):
            # doing the epoch
            train_logs = self.fit_epoch(dataset, n_batch=n_batch, shuffle=shuffle)
            if self.scheduler is not None:
                self.scheduler.step(train_logs['train_dice_loss'])

            if val_dataset is not None:
                val_logs = self.val_epoch(val_dataset, n_batch=n_batch, metric_list=metric_list)
                loss = val_logs['val_dice_loss']

            else:
                loss = train_logs['train_dice_loss']

            if save_model:
                # saving best model
                if loss < min_loss:
                    torch.save(self.net, os.path.join(self.checkpoint_folder, 'best_model.pt'))
                    min_loss = loss

                # saving latest model
                torch.save(self.net, os.path.join(self.checkpoint_folder, 'latest_model.pt'))

            # measuring time and memory
            epoch_end = time()
            # logging
            logs = {'epoch': epoch_idx,
                    'time': epoch_end - train_start,
                    **val_logs, **train_logs}

            with self.writer.as_default():
                tf.summary.scalar('time_per_epoch', epoch_end - train_start, step=epoch_idx)
                tf.summary.scalar('val_loss', val_logs['val_dice_loss'], step=epoch_idx)
                #tf.summary.scalar('val_iou_loss', val_logs['val_iou_loss'], step=epoch_idx)
                tf.summary.scalar('train_loss', train_logs['train_dice_loss'], step=epoch_idx)
                #tf.summary.scalar('train_iou_loss', train_logs['train_iou_loss'], step=epoch_idx)
            logger.log(logs)
            logger.to_csv(os.path.join(self.checkpoint_folder, 'logs.csv'))

            # saving model and logs
            if save_freq and (epoch_idx % save_freq == 0):
                epoch_save_path = os.path.join(self.checkpoint_folder, str(epoch_idx).zfill(4))
                chk_mkdir(epoch_save_path)
                torch.save(self.net, os.path.join(epoch_save_path, 'model.pt'))
                if predict_dataset:
                    self.predict_dataset(predict_dataset, epoch_save_path)

        # save the logger
        self.logger = logger

        return logger
    def predict_dataset(self, dataset, export_path):
        """
        Predicts the images in the given dataset and saves it to disk.

        Args:
            dataset: the dataset of images to be exported, instance of unet.dataset.Image2D
            export_path: path to folder where results to be saved
        """
        self.net.train(False)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device=self.device))
            y_out = self.net(X_batch).cpu().data.numpy()

            io.imsave(os.path.join(export_path, image_filename), y_out[0, 1, :, :].astype('uint8'))



