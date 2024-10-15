import numpy as np
from sklearn.model_selection import train_test_split

def sine_cosine_to_angle(sine_cosine):
    predicted_angles = np.empty((sine_cosine.shape[0], sine_cosine.shape[1], sine_cosine.shape[2]//2))
    for i in range(0, sine_cosine.shape[2], 2):
        sine = sine_cosine[:, :, i]
        cosine = sine_cosine[:, :, i + 1]
        angle = np.arctan2(sine, cosine)
        predicted_angles[:, :, i//2] = np.degrees(angle)
    return predicted_angles

def train_val_split(features, sine_cosine_values, attention_masks, pos_ids, test_size=0.2):
    return train_test_split(features, sine_cosine_values, attention_masks, pos_ids, test_size=test_size, random_state=0)
