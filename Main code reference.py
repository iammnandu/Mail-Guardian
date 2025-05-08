import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpamDetector:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        
    def preprocess_text(self, text):
        """Clean and preprocess the text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords and stem words
        tokens = []
        for word in text.split():
            if word not in self.stop_words:
                tokens.append(self.stemmer.stem(word))
        
        return ' '.join(tokens)
    
    def load_data(self, file_path=None):
        """
        Load the spam dataset. If file_path is not provided, 
        a sample dataset will be created for demonstration.
        """
        if file_path:
            # Load your own dataset
            data = pd.read_csv(file_path)
        else:
            # Create a small sample dataset for demonstration
            emails = [
                "Free viagra now!!! Click here for discount",
                "Congratulations! You've won a million dollars",
                "Meeting scheduled for tomorrow at 10am",
                "URGENT: Your account has been compromised",
                "Please review the attached document",
                "Get rich quick scheme, 100% guaranteed",
                "Your package will be delivered today",
                "Exclusive offer just for you, 80% off",
                "Weekly team meeting notes attached",
                "Password reset requested for your account"
            ]
            
            labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0]  # 1 for spam, 0 for ham
            
            data = pd.DataFrame({
                'text': emails,
                'label': labels
            })
        
        # Preprocess the text
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        return data
    
    def train_model(self, data, model_type='naive_bayes'):
        """Train the spam detection model."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed_text'], 
            data['label'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Create a pipeline
        if model_type == 'naive_bayes':
            self.model = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB())
            ])
        elif model_type == 'logistic_regression':
            self.model = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(max_iter=1000))
            ])
        elif model_type == 'svm':
            self.model = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SVC(probability=True))
            ])
        else:
            raise ValueError("Invalid model type. Choose from 'naive_bayes', 'logistic_regression', or 'svm'")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Return evaluation metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': X_test,
            'test_labels': y_test,
            'predictions': y_pred
        }
        
        return results
    
    def predict(self, text):
        """Predict if a given text is spam or not."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict([processed_text])[0]
        probability = self.model.predict_proba([processed_text])[0]
        
        result = {
            'text': text,
            'is_spam': bool(prediction),
            'confidence': probability[prediction],
            'processed_text': processed_text
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the spam detector
    detector = SpamDetector()
    
    # Load data (using sample data in this example)
    data = detector.load_data()
    
    print(f"Loaded {len(data)} emails")
    print(f"Spam emails: {sum(data['label'])}")
    print(f"Ham emails: {len(data) - sum(data['label'])}")
    
    # Train and evaluate models
    models = ['naive_bayes', 'logistic_regression', 'svm']
    best_model = None
    best_accuracy = 0
    
    for model_type in models:
        print(f"\nTraining {model_type} model...")
        results = detector.train_model(data, model_type)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Classification Report:")
        print(results['classification_report'])
        print("Confusion Matrix:")
        print(results['confusion_matrix'])
        
        if results['accuracy'] > best_accuracy:
            best_accuracy = results['accuracy']
            best_model = model_type
    
    print(f"\nBest model: {best_model} with accuracy {best_accuracy:.4f}")
    
    # Test with new emails
    test_emails = [
        "Congratulations! You've won our lottery! Send your bank details to claim your prize.",
        "Can we reschedule our meeting from 2pm to 3pm tomorrow?",
        "AMAZING OPPORTUNITY! Make $5000 from home with this simple trick!"
    ]
    
    print("\nTesting with new emails:")
    # Re-train with the best model
    detector.train_model(data, best_model)
    
    for email in test_emails:
        result = detector.predict(email)
        prediction = "SPAM" if result['is_spam'] else "HAM"
        print(f"\nEmail: {email}")
        print(f"Prediction: {prediction} (Confidence: {result['confidence']:.4f})")