import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

# Connect to MLflow Tracking server
mlflow.set_tracking_uri("http://192.168.1.60:5000")
mlflow.set_experiment("varun")

# Start an MLflow run
with mlflow.start_run():
    # Log params & metrics
    mlflow.log_param("n_estimators", 10)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)
    
    # Save model to artifact store
    mlflow.sklearn.log_model(clf, "model")
    
    # Generate Confusion Matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=load_iris().target_names, yticklabels=load_iris().target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save figure locally and log to MLflow
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print("✅ Model and graph logged to MLflow!")
