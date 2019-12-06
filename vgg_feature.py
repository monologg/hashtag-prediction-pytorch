import os
import sys

import torch
import numpy as np


def load_vgg_features(img_feature_file, img_ids):
    img_features = []
    for image_num in img_ids:
        feature = img_feature_file.get(image_num)
        if not feature:
            print("Errrrr")
        img_features.append(np.array(feature))
    img_tensor_features = torch.tensor(img_features, dtype=torch.float).squeeze(1)
    print(img_tensor_features.size())
    return img_tensor_features
