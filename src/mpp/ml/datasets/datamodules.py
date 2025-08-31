#standard library imports
import logging

#third party imports
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

#custom imports
from mpp.constants import VOCAB
from mpp.ml.datasets.fabricad import Fabricad
from mpp.ml.datasets.tkms import TKMS_Process_Dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")

def collate_fn(batch):
    """
    Custom collate function for batching variable-length target sequences.

    This function stacks input feature vectors and pads the target sequences
    (plans) to a fixed maximum length using the PAD token from the VOCAB.

    Parameters
    ----------
    batch : list of tuples
        Each element is a (vecset, plan) tuple where:
        - vecset : torch.Tensor of shape (set_size, input_dim)
        - plan : torch.Tensor of variable length (sequence of token indices)

    Returns
    -------
    vecsets : torch.Tensor
        Stacked feature vectors of shape (batch_size, set_size, input_dim).
    padded_plans : torch.Tensor
        Padded sequences of shape (batch_size, max_len), where padding tokens
        are added to match the maximum allowed sequence length.
    """
    vecsets, plans = zip(*batch)
    vecsets = torch.stack(vecsets)

    max_len = 10
    padded_plans = torch.full((len(plans), max_len), VOCAB["PAD"], dtype=torch.long)

    for i, plan in enumerate(plans):
        padded_plans[i, :plan.size(0)] = plan

    return vecsets, padded_plans

class MPP_datamodule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading the Fabricad or TKMS dataset.

    This module handles loading and batching of the
    dataset according to the specified input and target types.

    Parameters
    ----------
    batch_size : int, optional
        Batch size to be used in data loaders (default: 32).
    num_workers : int, optional
        Number of subprocesses to use for data loading (default: 4).
    input_type : str, optional
        Type of input to be used. Options include:
        - "vecset": for vector set input (e.g. CAD representations)
        - Other types may be supported depending on the dataset implementation.
    target_type : str, optional
        Type of target labels. Options include:
        - "seq": for step-by-step sequences
        - "class": for single-label classification
        - Others as defined in the specific dataset.
    """
    def __init__(self, batch_size=32, num_workers=4, input_type="vecset", target_type="seq", dataset="tkms"):
        super().__init__()
        logger.info("Initializing Fabricad datamodule")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_type=input_type
        self.target_type =target_type

        self.dataset = dataset

    def setup(self, stage=None):
        """
        Sets up datasets for different stages of training, validation, and testing.

        Parameters
        ----------
        stage : str or None
            One of 'fit', 'test', or None (default).
            If 'fit', initializes training and validation datasets.
            If 'test', initializes the test dataset.
            If None, initializes all datasets.

        Notes
        -----
        The setup uses the configured input and target types to load the data accordingly.
        These are passed directly to the dataset constructor (Fabricad or TKMS_Process_Dataset).
        """
        logger.info(f"Setting up  datamodule for stage: {stage} and Dataset {self.dataset}")
        if stage == "fit" or stage is None:
            if self.dataset == "fabricad":
                self.train_dataset = Fabricad(mode="train", input_type=self.input_type, target_type=self.target_type)
                self.val_dataset = Fabricad(mode="valid", input_type=self.input_type, target_type=self.target_type)
            elif self.dataset == "tkms":
                self.train_dataset = TKMS_Process_Dataset(mode="train", input_type=self.input_type, target_type=self.target_type)
                self.val_dataset = TKMS_Process_Dataset(mode="valid", input_type=self.input_type, target_type=self.target_type) 
            else:
                raise ValueError(f"Dataset {self.dataset} is not implemented")

            logger.info(f"Train dataset size: {len(self.train_dataset)}, Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            if self.dataset == "fabricad":
                self.test_dataset = Fabricad(mode="test", input_type=self.input_type, target_type=self.target_type)
            elif self.dataset == "tkms":
                self.test_dataset = TKMS_Process_Dataset(mode="test", input_type=self.input_type, target_type=self.target_type)
            else:
                raise ValueError(f"Dataset {self.dataset} is not implemented")
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        """
        Returns the training data loader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the training set.
        """
        logger.debug("Creating train dataloader")
        if self.target_type=="seq":
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns the validation data loader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the validation set.
        """
        logger.debug("Creating validation dataloader")
        if self.target_type=="seq":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self):
        """
        Returns the test data loader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the test set.
        """
        logger.debug("Creating test dataloader")
        if self.target_type=="seq":
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
        else:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



#checks
if __name__ == "__main__":
    
    # test Fabricad
    #for input_type in ["vecset"]:
    #    for target_type in ["time", "cost", "step-set", "seq"]:
    #        vecset_data_module = MPP_datamodule(batch_size=32, num_workers=4, input_type=input_type, target_type=target_type, dataset="fabricad")
    #        vecset_data_module.setup(stage="fit")
    #        
    #        train_loader = vecset_data_module.train_dataloader()
    #        validation_loader = vecset_data_module.val_dataloader()

    #        train_batch = next(iter(train_loader))
    #        validation_batch = next(iter(validation_loader))

    #        logger.info("Train batch shape:", train_batch[0].shape, train_batch[1].shape)
    #        logger.info("Validation batch shape:", validation_batch[0].shape, validation_batch[1].shape)

    # test for TMKS-data
    for input_type in ["vecset"]:
        for target_type in [ "step-set", "seq"]:
            vecset_data_module = MPP_datamodule(batch_size=32, num_workers=4, input_type=input_type, target_type=target_type, dataset="tkms")
            vecset_data_module.setup(stage="fit")
            
            train_loader = vecset_data_module.train_dataloader()
            validation_loader = vecset_data_module.val_dataloader()

            train_batch = next(iter(train_loader))
            validation_batch = next(iter(validation_loader))

            logger.info(f"Train INPUT/TARGET Shapes: {train_batch[0].shape} {train_batch[1].shape}")
            logger.info(f"Valid INPUT/TARGET Shapes: {validation_batch[0].shape}, {validation_batch[1].shape}")

