from tensorflow.keras.optimizers import Adam
from data_preprocessing import train_val_split
from model import get_model
from loss_functions import custom_loss_SAINT

def train_model(features, sine_cosine_values, attention_masks, pos_ids, num_features, MAXIMUM_LENGTH, epochs=120, batch_size=32):
    model = get_model(num_features, MAXIMUM_LENGTH)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_loss_SAINT)

    X_train, X_val, y_train, y_val, attn_mask_train, attn_mask_val, pos_ids_train, pos_ids_val = train_val_split(
        features, sine_cosine_values, attention_masks, pos_ids
    )

    history = model.fit(
        [X_train, attn_mask_train, pos_ids_train], y_train,
        validation_data=([X_val, attn_mask_val, pos_ids_val], y_val),
        epochs=epochs, 
        batch_size=batch_size
    )

    return model, history

def evaluate_model(model, X_val, attn_mask_val, pos_ids_val, y_val):
    predictions = model.predict([X_val, attn_mask_val, pos_ids_val])
    predicted_angles_degrees = sine_cosine_to_angle(predictions)
    true_angles_degrees = sine_cosine_to_angle(y_val)
    periodic_mae_value = periodic_mae(true_angles_degrees, predicted_angles_degrees)
    return periodic_mae_value
