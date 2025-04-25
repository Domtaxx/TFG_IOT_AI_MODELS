import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import joblib



# 🔹 Step 1: Load Train & Test Datasets
train_file = r"AI_models/Dataset/Data/FINAL_CSV/train70_augmented.csv"  # Update with your file path
test_file = r"AI_models/Dataset/Data/FINAL_CSV/test30_augmented.csv"  # Update with your file path

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# 🔹 Step 2: Drop Non-Numeric Features (Consistently for Both Datasets)
categorical_columns = ['tcp.flags', 'mqtt.conack.flags', 'mqtt.conflags', 'mqtt.hdrflags', 
                       'mqtt.msg', 'mqtt.protoname', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willtopic']

df_train = df_train.drop(columns=categorical_columns, errors='ignore')
#df_train = df_train.sample(frac=0.3, random_state=42)
df_test = df_test.drop(columns=categorical_columns, errors='ignore')

# 🔹 Step 3: Encode Target Labels Consistently
label_encoder = LabelEncoder()
df_train['target'] = label_encoder.fit_transform(df_train['target'])
df_test['target'] = label_encoder.transform(df_test['target'])  # Use the same encoding
joblib.dump(label_encoder, "AI_models/Random_forest/label_encoder.pkl")

# 🔹 Step 4: Separate Features (X) and Labels (y)
X_train = df_train.drop(columns=['target'])
y_train = df_train['target']

X_test = df_test.drop(columns=['target'])
y_test = df_test['target']

# 🔹 Step 5: Standardize Data (Use the Same Scaler for Consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply the same transformation
joblib.dump(scaler, "AI_models/Random_forest/scaler.pkl")
# 🔹 Step 6: Train a Random Forest Classifier
start = time.time()
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=8)
model.fit(X_train_scaled, y_train)
end = time.time()
print(f"Training Time: {end - start:.2f} seconds")

# 🔹 Step 7: Evaluate on Test Data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)


# Save the trained model
joblib.dump(model, "AI_models/Random_forest/random_forest_model.pkl")
print("Model saved successfully!")



# 🔹 Step 8: Print Results
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)


