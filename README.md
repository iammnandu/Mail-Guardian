# 📧 Spam Mail Detection using Machine Learning

A practical machine learning project that helps you identify whether an email or text message is spam or legitimate communication.

## 🎯 Project Goal
Build a machine learning model that can accurately classify messages as spam or not spam (ham) to help filter unwanted communications.

## 🛠️ Technologies Used
- Python 3.x
- Pandas & NumPy
- Matplotlib & Seaborn
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Google Colab

## 📊 Dataset
This project uses the UCI SMS Spam Collection dataset:
- [Kaggle: SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Contains labeled SMS messages (spam/ham)

## 📋 Project Steps

### 1️⃣ Environment Setup
```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2️⃣ Loading and Preparing Data
```python
# Load the dataset
df = pd.read_csv('/content/spam.csv', encoding='latin-1')

# Keep only necessary columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
```

### 3️⃣ Exploratory Data Analysis
```python
# Check dataset dimensions and info
df.shape
df.info()
df.isnull().sum()

# Analyze spam vs ham distribution
df['label'].value_counts()

# Visualize the distribution
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Mail Count")
plt.show()
```

### 4️⃣ Text Preprocessing
```python
# Import NLP tools
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Create text cleaning function
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()                    # Convert to lowercase
    text = text.split()                    # Split into words
    
    # Remove stopwords & apply stemming
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Apply cleaning to all messages
df['cleaned_message'] = df['message'].apply(clean_text)
```

### 5️⃣ Feature Engineering
```python
# Convert text to numerical features
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(df['cleaned_message']).toarray()

# Prepare target labels
y = pd.get_dummies(df['label'], drop_first=True)
```

### 6️⃣ Model Building and Training
```python
# Split data for training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

### 7️⃣ Model Evaluation
```python
# Evaluate model performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 8️⃣ Making Predictions
```python
# Function to predict new messages
def predict_mail(text):
    cleaned = clean_text(text)
    vector = cv.transform([cleaned]).toarray()
    result = model.predict(vector)
    return "Spam" if result[0]==1 else "Not Spam"

# Example usage
print(predict_mail("Congratulations! You won $10,000 lottery"))  # Spam
print(predict_mail("Hi, let's meet tomorrow for work."))         # Not Spam
```

### 9️⃣ Model Persistence
```python
# Save model for future use
import pickle

pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))
```

## 📊 Results
The model achieves good accuracy in distinguishing between spam and legitimate messages, making it effective for filtering unwanted communications.

## 🚀 How to Use
1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script
4. Use the `predict_mail()` function to classify new messages

## 🤝 Contributing
Contributions are welcome! Feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements
- UCI Machine Learning Repository for the dataset
- Kaggle for hosting the dataset
