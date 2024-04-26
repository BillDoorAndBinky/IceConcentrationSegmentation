import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet


class model_handler:
    RESIZE_SIZE = 448

    def __init__(self):
        # Определение функции потерь
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ENCODER = 'resnext50_32x4d'
        ENCODER_WEIGHTS = 'imagenet'
        CLASSES = ['high-concentration', 'ground', 'low-concentration', "water"]
        ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multiclass segmentation

        # Определение архитектуры U-Net

        self.model = Unet(encoder_name=ENCODER,
                          encoder_weights=ENCODER_WEIGHTS,
                          classes=len(CLASSES),
                          activation=ACTIVATION).to(device)

        self.model = torch.load('./resnext50_32x4d.pth', map_location=torch.device('cpu'))

    def predict(self, input):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_tensor = torch.from_numpy(input).to(device).unsqueeze(0)
        pr_mask = self.model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()

        for x in range(self.RESIZE_SIZE):
            for y in range(self.RESIZE_SIZE):
                maxx = pr_mask[0, x, y]
                maxq = 0
                for q in range(4):
                    if (pr_mask[q, x, y] >= maxx):
                        maxx = pr_mask[q, x, y]
                        maxq = q
                for q in range(4):
                    if (q == maxq):
                        pr_mask[q, x, y] = 1
                        continue
                    pr_mask[q, x, y] = 0

        return pr_mask
