import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import hamming_loss, classification_report, confusion_matrix

# Load your dataset (this should be replaced with your actual dataset loading code)
data = pd.read_csv("../data/final-combined-ds.csv")

# Feature and target variables
X = list(data["lyric"])
emotion_columns = ['anger', 'confidence', 'desire', 'disgust', 'gratitude',
                   'joy', 'love', 'lust',  'sadness', 'shame',
                   'fear',  'anticipation']
y = data[emotion_columns].values

# Split data into training and testing sets (this should be replaced with your actual split code)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1))
])

# Define the parameter grid
param_grid = {
    'tfidf__max_features': [100,150,200,250,300],  # Number of features for TfidfVectorizer
    'clf__estimator__n_estimators': [50, 100, 200],  # Number of trees in the forest
    'clf__estimator__max_depth': [5,10, 20, 30],  # Maximum depth of the tree
    # ... add other parameters here if needed
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Predictions using the best found parameters
y_pred = grid_search.predict(X_test)


print("Hamming Loss:", hamming_loss(y_test, y_pred))

report = classification_report(y_test, y_pred, target_names=emotion_columns, zero_division=0)
print(report)
# Compute and print confusion matrix for each emotion
for i, emotion in enumerate(emotion_columns):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    print(f"Confusion Matrix for {emotion}:\n{cm}\n")