import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os

# Create output directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the training data
print("Loading training data...")
df_train = pd.read_csv('data_train.csv')

print(f"Training dataset shape: {df_train.shape}")
print(f"\nColumn names: {df_train.columns.tolist()}")
print(f"\nTarget distribution:\n{df_train['ProdTaken'].value_counts()}")

# Separate features and target
X_train = df_train.drop('ProdTaken', axis=1)
y_train = df_train['ProdTaken']

# Handle categorical variables using one-hot encoding
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

# One-hot encode categorical features
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)

print(f"Features after encoding: {X_train_encoded.shape[1]}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")

# Build an improved neural network with better architecture
print("\nBuilding neural network model...")
model = keras.Sequential([
    # Input layer + First hidden layer
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    # Second hidden layer
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    # Third hidden layer
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    # Fourth hidden layer
    keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.1),

    # Output layer
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with adjusted learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Display model summary
model.summary()

# Train the model with more epochs
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
)



# Save the model
model.save('models/travel_classification_model.h5')
print("\nModel saved to models/travel_classification_model.h5")

# Save the scaler for later use in inference
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to models/scaler.pkl")

# Save the feature column names for inference
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X_train_encoded.columns.tolist(), f)
print("Feature columns saved to models/feature_columns.pkl")
