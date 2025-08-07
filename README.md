# 📚 Book Recommendation Engine with KNN

A simple yet powerful book recommendation app that suggests similar books using collaborative filtering and the K-Nearest Neighbors (KNN) algorithm.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arib-book-recommendation.streamlit.app/)

## 🔍 How It Works

This app uses **item-based collaborative filtering** to recommend books based on user rating patterns from the [Book-Crossing Dataset](https://www.bookcrossing.com/). When you search for a book, the system:

1. Finds books with similar user rating patterns
2. Uses **cosine similarity** and **KNN** to identify the most similar books
3. Returns recommendations with similarity scores

## 🚀 Features

- Search for any book in the dataset
- Fuzzy matching to handle partial/misspelled titles
- Top-N recommendations with similarity scores
- Clean, intuitive interface

## 🛠️ Technical Details

- **Algorithm**: K-Nearest Neighbors (KNN) with cosine similarity
- **Data**: Book-Crossing Dataset (filtered to active users and popular books)
- **Stack**: Python, Streamlit, scikit-learn, pandas
- **Optimization**: Pre-processed dataset stored on GitHub for faster loading

