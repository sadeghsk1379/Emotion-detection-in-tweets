Sentiment Analysis using SVM
This project performs sentiment analysis on textual data using Support Vector Machines (SVM) and TF-IDF vectorization.

Overview
The project involves:

Data Handling: Reading and preprocessing textual data from data.xlsx.
Model Training: Using SVM with TF-IDF for feature extraction and sentiment classification.
Evaluation: Assessing model performance with accuracy metrics and a detailed classification report.
Prediction: Applying the trained model to new data (new_reviews.xlsx) to predict sentiment and calculate scores.
Features
SVM Classifier: Utilizes SVM for sentiment classification.
TF-IDF Vectorization: Extracts features from text data.
Data Preprocessing: Cleans and prepares text data for analysis.
Evaluation Metrics: Calculates accuracy and provides detailed classification reports.
Libraries Used
pandas: Data manipulation and analysis.
sklearn: Machine learning library including model selection, feature extraction, SVM model, and metrics.
nltk: Natural Language Toolkit for stop words.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/sentiment-analysis-svm.git
cd sentiment-analysis-svm
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Ensure you have the necessary data files:

Place data.xlsx and new_reviews.xlsx in the project directory.
Running the Project
Run the sentiment analysis script:

bash
Copy code
python sentiment_analysis.py
View the output:

The script will display accuracy, classification report, and sentiment predictions for new reviews.
Example Output
Accuracy: Shows the accuracy of the SVM model on the test data.
Classification Report: Provides precision, recall, and F1-score for each sentiment class.
Sentiment Predictions: Displays the number of positive and negative reviews and calculates a score out of 10 based on positive reviews.