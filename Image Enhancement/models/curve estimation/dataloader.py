import os
import random
from typing import List
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob

class LowlightDataset(data.Dataset):
    def __init__(self, image_dir: str, image_size: int = 256) -> None:
        self.image_paths = self._collect_images(image_dir)
        self.image_size = image_size
        print(f"Total training samples: {len(self.image_paths)}")

    @staticmethod
    def _collect_images(image_dir: str) -> List[str]:
        images = glob.glob(os.path.join(image_dir, "*.jpg"))
        random.shuffle(images)
        return images

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        image = Image.open(image_path).resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image_tensor

    def __len__(self) -> int:
        return len(self.image_paths)
