import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle

# Create output directory if it doesn't exist
os.makedirs('output5', exist_ok=True)

# Load the trained model
print("Loading trained model...")
model = keras.models.load_model('models/travel_classification_model.h5')
print("Model loaded successfully!")

# Display model summary
model.summary()

# Load the scaler and feature columns
print("\nLoading scaler and feature information...")
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

print("Scaler and feature columns loaded!")

# Load the test data
print("\nLoading test data...")
df_test = pd.read_csv('data_test.csv')

print(f"Test dataset shape: {df_test.shape}")
print(f"\nTarget distribution:\n{df_test['ProdTaken'].value_counts()}")

# Separate features and target
X_test = df_test.drop('ProdTaken', axis=1)
y_test = df_test['ProdTaken']

# Handle categorical variables using one-hot encoding
# Must match the training data preprocessing
categorical_columns = X_test.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

# One-hot encode categorical features
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

# Ensure all training feature columns are present
for col in feature_columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0

# Reorder columns to match training data
X_test_encoded = X_test_encoded[feature_columns]

print(f"Features after encoding: {X_test_encoded.shape[1]}")

# Standardize the features using the loaded scaler
X_test_scaled = scaler.transform(X_test_encoded)

# Make predictions
print("\nMaking predictions...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate metrics
print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

try:
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
except:
    print("AUC Score: Could not calculate")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Product', 'Product Taken']))

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Product', 'Product Taken'],
            yticklabels=['No Product', 'Product Taken'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Travel Product Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add additional statistics as text
tn, fp, fn, tp = cm.ravel()
stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
stats_text += f'\n\nAccuracy: {accuracy:.4f}'
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()

# Save the figure
output_path = 'output5/confusion_matrix.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix plot saved to: {output_path}")

# Also create a normalized confusion matrix
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['No Product', 'Product Taken'],
            yticklabels=['No Product', 'Product Taken'],
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix - Travel Product Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
output_path_normalized = 'output5/confusion_matrix_normalized.jpg'
plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")

# Save predictions to CSV
results_df = df_test.copy()
results_df['predicted_label'] = y_pred
results_df['predicted_label_name'] = results_df['predicted_label'].map({0: 'No Product', 1: 'Product Taken'})
results_df['prediction_probability'] = y_pred_proba.flatten()
results_df['actual_label_name'] = results_df['ProdTaken'].map({0: 'No Product', 1: 'Product Taken'})
results_df['correct_prediction'] = (results_df['ProdTaken'] == results_df['predicted_label'])

output_csv_path = 'output5/predictions.csv'
results_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")

print("\n" + "="*60)
print("Inference completed successfully!")
print("="*60)
