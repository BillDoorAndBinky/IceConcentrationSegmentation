import os
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu

class image_preprocessor:
    RESIZE_SIZE = 448
    ENCODER = 'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'

    def preprocess(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.RESIZE_SIZE, self.RESIZE_SIZE))
        image = np.array(image)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.ENCODER, self.ENCODER_WEIGHTS)

        if preprocessing_fn:
            sample = self.get_preprocessing(preprocessing_fn)(image=image)
            image = sample['image']

        return image

    def get_preprocessing(self, pf):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=pf),
            albu.Lambda(image=self.to_tensor),
        ]
        return albu.Compose(_transform)

    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
