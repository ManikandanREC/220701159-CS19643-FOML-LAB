import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_csv('HAM10000_metadata.csv')
df = df[['age', 'sex', 'localization', 'dx']]
df.dropna(inplace=True)

# Encode categorical data
le_sex = LabelEncoder()
le_loc = LabelEncoder()
le_dx = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['localization'] = le_loc.fit_transform(df['localization'])
df['dx'] = le_dx.fit_transform(df['dx'])

# Diagnosis label mapping
label_map = dict(zip(le_dx.transform(le_dx.classes_), le_dx.classes_))
full_names = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# Features and target
X = df[['age', 'sex', 'localization']]
y = df['dx']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SVM Model ---
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# --- Random Forest Model ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Show model accuracies
print(f"SVM Accuracy: {svm_acc*100:.2f}%")
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# Predict for a sample
sample = pd.DataFrame([[45, le_sex.transform(['female'])[0], le_loc.transform(['upper extremity'])[0]]], columns=X.columns)

# Decode predictions
svm_prediction = le_dx.inverse_transform(svm_model.predict(sample))[0]
rf_prediction = le_dx.inverse_transform(rf_model.predict(sample))[0]

print("\nPredicted diagnosis for sample (SVM):", full_names[svm_prediction])
print("Predicted diagnosis for sample (Random Forest):", full_names[rf_prediction])


y_test_labels = le_dx.inverse_transform(y_test)
svm_pred_labels = le_dx.inverse_transform(svm_pred)
rf_pred_labels = le_dx.inverse_transform(rf_pred)

# Create DataFrame for plotting
df_compare = pd.DataFrame({
    'Actual': y_test_labels,
    'SVM': svm_pred_labels,
    'Random Forest': rf_pred_labels
})

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# SVM
sns.countplot(x='Actual', hue='SVM', data=df_compare, ax=axes[0])
axes[0].set_title("SVM: Actual vs Predicted")
axes[0].tick_params(axis='x', rotation=45)

# Random Forest
sns.countplot(x='Actual', hue='Random Forest', data=df_compare, ax=axes[1])
axes[1].set_title("Random Forest: Actual vs Predicted")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
