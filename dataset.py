
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class PatchDataset(Dataset):
    def __init__(self, coords_df, counts, transform, is_train=False, cache_root='./'):
        super().__init__()
        self.coords_df = coords_df
        self.counts = counts

        self.transform = transform 
        self.is_train = is_train
        self.cache_root = cache_root

    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, idx): 
        patch = Image.open(os.path.join(self.cache_root, self.coords_df.loc[idx, 'patch_filename']))
        if self.is_train:
            if np.random.rand() < 0.5:
                patch = patch.rotate(np.random.choice([90, 180, 270]))
            # if np.random.rand() < 0.2:
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=np.random.randint(low=1,high=50)/100.)) 
        label = self.counts[idx]
        return self.transform(patch), label
















