import numpy as np
import random

# class Simple2DTransform:
#     def __init__(self, flip_prob=0.5):
#         """
#         Minimal 2D transform: Random horizontal flip.
#         """
#         self.flip_prob = flip_prob

#     def __call__(self, sample):
#         image, label = sample['image'], sample.get('label', None)

#         # === Random Horizontal Flip ===
#         if random.random() < self.flip_prob:
#             image = np.flip(image, axis=2).copy()   # Flip width
#             if label is not None:
#                 label = np.flip(label, axis=1).copy()

#         return {'image': image, 'label': label} if label is not None else {'image': image}


class Simple2DTransform:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, label=None):
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=2).copy()
            if label is not None:
                label = np.flip(label, axis=1).copy()
        return image, label
