import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import re

# خواندن داده‌ها از فایل اکسل
df = pd.read_excel('data.xlsx')

# نمایش چند خط اول داده برای بررسی
print(df.head())

# پیش‌پردازش متن (حذف علائم نگارشی و تبدیل به حروف کوچک)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # حذف علائم نگارشی
    return text

df['text'] = df['text'].apply(preprocess_text)

# تقسیم داده به مجموعه آموزش و آزمایش
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استفاده از TF-IDF برای استخراج ویژگی‌ها از متن
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ساخت مدل SVM برای طبقه‌بندی
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# پیش‌بینی بر روی داده‌های تست
y_pred = svm_model.predict(X_test_tfidf)

# ارزیابی مدل
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# خواندن داده‌های جدید (نظرات کاربران در مورد محصول)
new_reviews = pd.read_excel('new_reviews.xlsx')

# پیش‌پردازش داده‌های جدید
new_reviews['text'] = new_reviews['text'].apply(preprocess_text)

# استخراج ویژگی‌ها از داده‌های جدید با استفاده از مدل TF-IDF
new_reviews_tfidf = vectorizer.transform(new_reviews['text'])

# پیش‌بینی بر روی داده‌های جدید
new_predictions = svm_model.predict(new_reviews_tfidf)

# تعداد نظرات مثبت و منفی
positive_reviews = sum(new_predictions)
negative_reviews = len(new_predictions) - positive_reviews

# محاسبه نمره از 10
total_reviews = len(new_predictions)
score = (positive_reviews / total_reviews) * 10

print(f'Total Reviews: {total_reviews}')
print(f'Positive Reviews: {positive_reviews}')
print(f'Negative Reviews: {negative_reviews}')
print(f'Score: {score:.2f} out of 10')
