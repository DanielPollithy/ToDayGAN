"""Assemble image retrieval network, with intermediate endpoints.

From: https://github.com/germain-hug/S2DHM
"""
from collections import OrderedDict
from .images_from_list import ImagesFromList
from .netvlad import NetVLAD
from tqdm import tqdm
from typing import List
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.functional import interpolate
from torchvision import models


class ImageRetrievalModel():
    """Build the image retrieval model with intermediate feature extraction.

    The model is made of a VGG-16 backbone combined with a NetVLAD pooling
    layer.
    """
    def __init__(self, checkpoint_path: str, device,
                 num_clusters=64, encoder_dim=512):
        """Initialize the Image Retrieval Network.

        Args:
            num_clusters: Number of NetVLAD clusters (should match pre-trained)
                weights.
            encoder_dim: NetVLAD encoder dimension.
            checkpoint_path: Path to the pre-trained weights.
            hypercolumn_layers: The hypercolumn layer indices used to compute
                the intermediate features.
            device: The pytorch device to run on.
        """
        self._num_clusters = num_clusters
        self._encoder_dim = encoder_dim
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model = self._build_model()

    def _build_model(self):
        """ Build image retrieval network and load pre-trained weights.
        """
        model = nn.Module()

        # Assume a VGG-16 backbone
        encoder = models.vgg16(pretrained=False)
        layers = list(encoder.features.children())[:-2]
        encoder = nn.Sequential(*layers)
        model.add_module('encoder', encoder)

        # Assume a NetVLAD pooling layer
        net_vlad = NetVLAD(
            num_clusters=self._num_clusters, dim=self._encoder_dim)
        model.add_module('pool', net_vlad)

        # For parallel training
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            model.encoder = DataParallel(model.encoder)
            model.pool = DataParallel(model.pool)

        # Load weights
        checkpoint = torch.load(self._checkpoint_path,
                                map_location=lambda storage,
                                loc: storage)['state_dict']

        # If model was not trained in parallel, adapt the keys
        if 'module' not in list(checkpoint.keys())[0] and False:
            checkpoint = OrderedDict((k.replace('encoder', 'encoder.module'), v)
                for k, v in checkpoint.items())
            checkpoint = OrderedDict((k.replace('pool', 'pool.module'), v)
                for k, v in checkpoint.items())
        elif gpu_count <=1:
            checkpoint = OrderedDict((k.replace('encoder.module', 'encoder'), v)
                for k, v in checkpoint.items())
            checkpoint = OrderedDict((k.replace('pool.module', 'pool'), v)
                for k, v in checkpoint.items())


        model.load_state_dict(checkpoint)
        if self._device != -1:
            model = model.to(self._device)
        model.eval()
        return model

    def compute_embedding(self, images):
        """Compute global image descriptor.

        Args:
            images: A list of image filenames.
            image_size: The size of the images to use.
            preserve_ratio: Whether the image ratio is preserved when resizing.
        Returns:
            descriptors: The global image descriptors, as numpy objects.
        """
        image_size = 512
        preserve_ratio = True
        # Build dataloader
        dataloader = ImagesFromList(images, image_size,
                                    preserve_ratio=preserve_ratio)
        # Compute descriptors
        if self._device != -1:
            with torch.no_grad():
                db_desc = torch.zeros((len(dataloader),
                    self._encoder_dim * self._num_clusters)).to(self._device)
                for i, tensor in enumerate(dataloader):
                    tensor = tensor.to(self._device).unsqueeze(0)
                    db_desc[i,:] = self._model.pool(self._model.encoder(tensor))
        else:
            with torch.no_grad():
                db_desc = torch.zeros((len(dataloader),
                    self._encoder_dim * self._num_clusters))
                for i, tensor in enumerate(dataloader):
                    tensor = tensor.unsqueeze(0)
                    db_desc[i,:] = self._model.pool(self._model.encoder(tensor))
        return db_desc.cpu().detach().numpy()

    @property
    def device(self):
        return self._device
