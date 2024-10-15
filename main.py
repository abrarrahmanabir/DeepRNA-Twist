import numpy as np
from training import train_model, evaluate_model
from data_preprocessing import sine_cosine_to_angle
from sklearn.preprocessing import StandardScaler

# Load your data here
features = np.load('path_to_features.npy')  # Load the features
sine_cosine_values = np.load('path_to_sine_cosine_values.npy')  # Load the sine and cosine values


def train_and_evaluate():
    model, history = train_model(
        features=features,
        sine_cosine_values=sine_cosine_values,
        attention_masks=attention_masks,
        pos_ids=pos_ids,
        num_features=features.shape[2],
        MAXIMUM_LENGTH=features.shape[1],
        epochs=120,
        batch_size=32
    )

    X_val = features[:int(0.2 * len(features))]
    attn_mask_val = attention_masks[:int(0.2 * len(attention_masks))]
    pos_ids_val = pos_ids[:int(0.2 * len(pos_ids))]
    y_val = sine_cosine_values[:int(0.2 * len(sine_cosine_values))]

    periodic_mae_value = evaluate_model(model, X_val, attn_mask_val, pos_ids_val, y_val)
    print(f"Total Mean Absolute Error: {periodic_mae_value} degrees")

if __name__ == '__main__':
    train_and_evaluate()
