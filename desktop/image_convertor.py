import numpy as np
from PIL import Image

class image_convertor:
    RESIZE_SIZE = 448

    color_palette = [
        [0, 191, 255],  # Класс 0 - Голубой (высокая сплоченность льда)
        [139, 69, 19],  # Класс 1 - Коричневый (земля)
        [65, 105, 225],  # Класс 2 - Светло-голубой (низкая сплоченность льда)
        [0, 0, 255],  # Класс 3 - Синий (вода)
        [0, 0, 0]  # Класс 4 - НЕИЗВЕСТНО
    ]

    def mask_to_image(self, mask):
        img = np.zeros((self.RESIZE_SIZE, self.RESIZE_SIZE, 3), dtype=np.uint8)

        for i in range(self.RESIZE_SIZE):
            for j in range(self.RESIZE_SIZE):
                found = False
                for q in range(4):
                    if (mask[q, i, j] == 1):
                        img[i, j] = self.color_palette[q]
                        found = True
                        break

                if (not found):
                    img[i, j] = self.color_palette[4]

        ouput = Image.fromarray(img)
        return ouput