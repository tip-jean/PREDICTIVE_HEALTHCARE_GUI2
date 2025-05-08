# For training a Random Forest model to predict diseases based on symptoms
# using the dataset 'symbipredict_2022.csv'

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
import joblib

df = pd.read_csv('dataset/symbipredict_2022.csv')


def train_model(df: pd.DataFrame) -> tuple:
    '''
    Trains a Random Forest model to predict diseases based on symptoms.
    Args:
        df (pd.DataFrame): DataFrame containing the dataset with symptoms and prognosis.
    Returns:
        (best_model, le) (tuple): The trained model and the label encoder.
    '''

    X = df.drop('prognosis', axis=1)

    y = df['prognosis']

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

    print(f'CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})')

    best_model.fit(X_train, y_train)

    accuracy = best_model.score(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    y_pred = best_model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=le.inverse_transform(sorted(set(y_test))))

    joblib.dump(best_model, 'models/disease_prediction_model.pkl')
    joblib.dump(le, 'encoders/label_encoder.pkl')

    return best_model, le


(best_model, le) = train_model(df)

print(f"Model: {best_model}\nLabel Encoder: {le}")
print("Model and Label Encoder have been trained and saved successfully.")
