import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading csv. File
df = pd.read_csv("C:\\Users\\user 123\\OneDrive\\Documents\\GitHub\\project\\project1\\project1Data.csv")
print(df.head())

# Step 2: Data Visualization

# 2.1 Visualizing the dataset behavior comparing X, Y, Z to Maintainence Step using histograms
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Define step values and plot histograms for X, Y, and Z
step_values = np.unique(df['Step'].values)

# Histogram for X variable grouped by Step
for step_value in step_values:
    x_values = df['X'].values[df['Step'].values == step_value]
    axs[0].hist(x_values, bins=20, alpha=0.5, label=f'Step {step_value}')
axs[0].set_title('X Variable Distribution by Step')
axs[0].set_xlabel('X Value')
axs[0].set_ylabel('Frequency')
axs[0].legend(title='Step')

# Histogram for Y variable grouped by Step
for step_value in step_values:
    y_values = df['Y'].values[df['Step'].values == step_value]
    axs[1].hist(y_values, bins=20, alpha=0.5, label=f'Step {step_value}')
axs[1].set_title('Y Variable Distribution by Step')
axs[1].set_xlabel('Y Value')
axs[1].set_ylabel('Frequency')
axs[1].legend(title='Step')

# Histogram for Z variable grouped by Step
for step_value in step_values:
    z_values = df['Z'].values[df['Step'].values == step_value]
    axs[2].hist(z_values, bins=20, alpha=0.5, label=f'Step {step_value}')
axs[2].set_title('Z Variable Distribution by Step')
axs[2].set_xlabel('Z Value')
axs[2].set_ylabel('Frequency')
axs[2].legend(title='Step')

# Adjust layout and show histograms
plt.tight_layout()
plt.show()

# 2.2 3D Scatter plot of X, Y, Z Variables by Step
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='coolwarm', s=100)
ax.set_xlabel('X Variable')
ax.set_ylabel('Y Variable')
ax.set_zlabel('Z Variable')
ax.set_title('3D Scatter Plot of (X, Y, Z) Variables by Step')
cbar = fig.colorbar(scat, ax=ax)
cbar.set_label('Step')
plt.show()

# Step 3: Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
print(correlation_matrix)

# Step 4: Classification Model Development
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score

# Splitting dataset into features (X, Y, Z) and target (Step)
X_features = df[['X', 'Y', 'Z']]
y_target = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

# Hyperparameter grids for GridSearchCV
param_grids = {
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7]}
}

# Perform GridSearchCV for each classifier
best_models = {}
for clf_name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[clf_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[clf_name] = grid_search.best_estimator_

# Evaluate models on test set
performance_metrics = {}
for clf_name, model in best_models.items():
    y_pred = model.predict(X_test)
    
    performance_metrics[clf_name] = {
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Accuracy': accuracy_score(y_test, y_pred)
    }
    
# Print the performance metrics for each model
print("Performance Metrics for Each Model:")
for clf_name, metrics in performance_metrics.items():
    print(f"\nModel: {clf_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
        
print(performance_metrics)

# Step 5: Model Performance Analysis with Confusion Matrix and Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Using SVM for confusion matrix
best_model_svm = best_models['SVM']
y_pred_svm = best_model_svm.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Plot confusion matrix with precision and accuracy
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
accuracy_svm = accuracy_score(y_test, y_pred_svm)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(conf_matrix).plot(ax=ax, cmap='Blues')
plt.text(0.1, -0.15, f'Precision: {precision_svm:.4f}', ha='center', transform=ax.transAxes, fontsize=12)
plt.text(0.9, -0.15, f'Accuracy: {accuracy_svm:.4f}', ha='center', transform=ax.transAxes, fontsize=12)
plt.title('Confusion Matrix for SVM Model with Precision and Accuracy')
plt.show()

# Step 6: Stacked Model Performance Analysis
from sklearn.ensemble import StackingClassifier

# Define stacking model (Random Forest and SVM)
estimators = [
    ('rf', best_models['Random Forest']),
    ('svm', best_models['SVM'])
]
stacked_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=42))
stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)

# Evaluate stacked model
stacked_performance = {
    'F1 Score': f1_score(y_test, y_pred_stacked, average='weighted'),
    'Precision': precision_score(y_test, y_pred_stacked, average='weighted'),
    'Accuracy': accuracy_score(y_test, y_pred_stacked)
}

# Print the performance metrics for the stacked model
print("Stacked Model Performance:")
for metric, value in stacked_performance.items():
    print(f"{metric}: {value:.4f}")
    
    
print("Performance of Stacked Model:")
print(stacked_performance)

# Confusion Matrix for Stacked Model
conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(conf_matrix_stacked).plot(ax=ax, cmap='Purples')
plt.text(0.1, -0.15, f'Precision: {stacked_performance["Precision"]:.4f}', ha='center', transform=ax.transAxes, fontsize=12)
plt.text(0.9, -0.15, f'Accuracy: {stacked_performance["Accuracy"]:.4f}', ha='center', transform=ax.transAxes, fontsize=12)
plt.title('Confusion Matrix for Stacked Model')
plt.show()

# Step 7: Save and Predict using Stacked Model
import joblib as jb

# Save the model
jb.dump(stacked_model, 'best_model_stacked.pkl')

# Load the model for prediction
loaded_model = jb.load('best_model_stacked.pkl')

# Predict on new data points
Provided_data = np.array([[9.375, 3.0625, 1.51], 
                          [6.995, 5.125, 0.3875], 
                          [0, 3.0625, 1.93], 
                          [9.4, 3, 1.8], 
                          [9.4, 3, 1.3]])

predictions = loaded_model.predict(Provided_data)
print(f"Predicted Corresponding Maintenance Steps for the Provided data points: {predictions}")
