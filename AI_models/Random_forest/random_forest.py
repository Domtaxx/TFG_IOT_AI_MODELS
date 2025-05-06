import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import joblib

# 🔹 Step 1: Load Train & Test Datasets
train_file = r"AI_models/Dataset/training/filtered_output_normal.csv"
test_file = r"AI_models/Dataset/testing/filtered_output_normal.csv"
columns_names = ["tcp.flags","tcp.time_delta","tcp.len","mqtt.conack.flags","mqtt.conack.flags.reserved","mqtt.conack.flags.sp","mqtt.conack.val","mqtt.conflag.cleansess","mqtt.conflag.passwd","mqtt.conflag.qos","mqtt.conflag.reserved","mqtt.conflag.retain","mqtt.conflag.uname","mqtt.conflag.willflag","mqtt.conflags","mqtt.dupflag","mqtt.hdrflags","mqtt.kalive","mqtt.len","mqtt.msg","mqtt.msgid","mqtt.msgtype","mqtt.proto_len","mqtt.protoname","mqtt.qos","mqtt.retain","mqtt.sub.qos","mqtt.suback.qos","mqtt.ver","mqtt.willmsg","mqtt.willmsg_len","mqtt.willtopic","mqtt.willtopic_len","target"]
expected_columns = ["tcp.time_delta","tcp.len","mqtt.conack.flags.reserved","mqtt.conack.flags.sp","mqtt.conack.val","mqtt.conflag.cleansess","mqtt.conflag.passwd","mqtt.conflag.qos","mqtt.conflag.reserved","mqtt.conflag.retain","mqtt.conflag.uname","mqtt.conflag.willflag","mqtt.dupflag","mqtt.kalive","mqtt.len","mqtt.msgid","mqtt.msgtype","mqtt.proto_len","mqtt.qos","mqtt.retain","mqtt.sub.qos","mqtt.suback.qos","mqtt.ver","mqtt.willmsg","mqtt.willmsg_len","mqtt.willtopic","mqtt.willtopic_len"]
df_train = pd.read_csv(train_file,names=columns_names, header=0)

df_test = pd.read_csv(test_file, names=columns_names, header=0)
# 🔹 Step 2: Prepare Features
important_text_cols = []  # Columns you want to keep

# Final feature set: numeric columns + important text columns
final_features = expected_columns

# Keep only the needed columns + target
df_train = df_train[final_features + ["target"]]
df_test = df_test[final_features + ["target"]]

# 🔹 Step 3: Separate Features (X) and Target (y)
X_train = df_train.drop(columns=["target"])
X_train = X_train[expected_columns]
y_train = df_train["target"]

X_test = df_test.drop(columns=["target"])
X_test = X_test[expected_columns]
y_test = df_test["target"]

# 🔹 Step 4: Encode Target Labels (NOW)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Save the label encoder (AFTER fitting)
joblib.dump(label_encoder, "AI_models/Random_forest/label_encoder.pkl")

# 🔹 Step 5: Scale Only Numeric Features
scaler = StandardScaler()
X_train_numeric = X_train[expected_columns]
X_test_numeric = X_test[expected_columns]

X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
X_test_numeric_scaled = scaler.transform(X_test_numeric)

# Save the scaler (AFTER fitting)
joblib.dump(scaler, "AI_models/Random_forest/scaler.pkl")

# 🔹 Step 6: Rebuild Feature Set
X_train_numeric_scaled_df = pd.DataFrame(X_train_numeric_scaled, columns=expected_columns)
X_test_numeric_scaled_df = pd.DataFrame(X_test_numeric_scaled, columns=expected_columns)

# Combine scaled numeric data with untouched important text columns
X_train_ready = pd.concat([X_train_numeric_scaled_df.reset_index(drop=True), X_train[important_text_cols].reset_index(drop=True)], axis=1)
X_test_ready = pd.concat([X_test_numeric_scaled_df.reset_index(drop=True), X_test[important_text_cols].reset_index(drop=True)], axis=1)

# 🔹 Step 7: Train the Random Forest
class_weights = {
    0: 2.0,  # dos
    1: 1.0,  # legitimate
    2: 10.0, # malformed
    3: 10.0, # slowite
}
start = time.time()
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=8, class_weight = class_weights)
model.fit(X_train_ready, y_train_encoded)
end = time.time()
print(f"Training Time: {end - start:.2f} seconds")

# 🔹 Step 8: Evaluate the Model
y_pred = model.predict(X_test_ready)
accuracy = accuracy_score(y_test_encoded, y_pred)
classification_rep = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)

# Save the trained Random Forest model
joblib.dump(model, "AI_models/Random_forest/random_forest_model.pkl")
print("Model saved successfully!")

# 🔹 Step 9: Print Results
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)
