import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.metrics import hamming_loss, classification_report, confusion_matrix


# Load your dataset
data = pd.read_csv("../data/final-combined-ds.csv")

# Feature and target variables
X = list(data["lyric"])
emotion_columns = ['anger', 'confidence', 'desire', 'disgust', 'gratitude',
                   'joy', 'love', 'lust',  'sadness', 'shame',
                   'fear',  'anticipation']
y = data[emotion_columns].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Classifier Chain with Naive Bayes
classifier_chain = ClassifierChain(MultinomialNB())


# Create a Pipeline
pipeline = Pipeline([
    ('vectorizer', tfidf),
    ('classifier', classifier_chain)
])

# Define the parameter grid
param_grid = {
    'vectorizer__max_features': [400,450,500,525,550,600,650,700,750,800,850,900,950,1000],
    'classifier__base_estimator__alpha': [0.01, 0.1, 1.0] # smoothing parameter of Naive Bayes
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate on the test set
y_pred = grid_search.predict(X_test)

# Evaluate the model
print("Hamming Loss:", hamming_loss(y_test, y_pred))

report = classification_report(y_test, y_pred, target_names=emotion_columns, zero_division=0)

print(report)

# Compute and print confusion matrix for each emotion
for i, emotion in enumerate(emotion_columns):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    print(f"Confusion Matrix for {emotion}:\n{cm}\n")

