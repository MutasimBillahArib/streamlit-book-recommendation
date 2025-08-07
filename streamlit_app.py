import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import requests
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Book Recommendation Engine",
    page_icon="📚",
    layout="centered"
)

@st.cache_resource
def load_data_and_model():
    """Load pre-processed data from GitHub"""
    # URL to your raw GitHub file (using raw.githubusercontent.com)
    GITHUB_RAW_URL = "https://github.com/MutasimBillahArib/machine-learning-with-python-freecodecamp/raw/refs/heads/main/book-recommendation-engine/book_recommendation_data.pkl"
    
    try:
        # Download the pre-processed data
        response = requests.get(GITHUB_RAW_URL)
        response.raise_for_status()
        
        # Verify we got binary data, not HTML
        if response.content.startswith(b'<'):
            st.error("Error: Downloaded content appears to be HTML, not pickle data. Check your GitHub URL.")
            st.stop()
            
        # Load the data
        data = pickle.loads(response.content)
        
        # Extract components
        title_to_isbn = data['title_to_isbn']
        isbn_to_title = data['isbn_to_title']
        df_pivot = data['df_pivot']
        
        # Build the sparse matrix and KNN model
        book_matrix = csr_matrix(df_pivot.values)
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(book_matrix)
        
        return title_to_isbn, isbn_to_title, book_matrix, model_knn, df_pivot
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

def get_recommends(book, title_to_isbn, isbn_to_title, book_matrix, model_knn, df_pivot, n_neighbors=6):
    """Get book recommendations"""
    if book not in title_to_isbn:
        return [book, []]

    isbn = title_to_isbn[book]
    if isbn not in df_pivot.index:
        return [book, []]

    idx = df_pivot.index.get_loc(isbn)
    distances, indices = model_knn.kneighbors(book_matrix[idx], n_neighbors=n_neighbors)

    # Skip the first result (the book itself)
    distances = distances.flatten()[1:]
    indices = indices.flatten()[1:]

    # Reverse to get most similar first
    distances = distances[::-1]
    indices = indices[::-1]

    recommended_books = []
    for i in range(len(indices)):
        book_isbn = df_pivot.index[indices[i]]
        title = isbn_to_title[book_isbn]
        dist = distances[i]
        recommended_books.append([title, dist])

    return [book, recommended_books]

def main():
    st.title("📚 Book Recommendation Engine")
    st.write("Find books similar to your favorite titles")
    
    try:
        # Load data and model
        title_to_isbn, isbn_to_title, book_matrix, model_knn, df_pivot = load_data_and_model()
        all_titles = list(title_to_isbn.keys())
        
        # Book search with suggestions
        st.subheader("Search for a book")
        search_term = st.text_input("Enter book title", "")
        
        if search_term:
            # Get top 10 matching titles
            matches = process.extract(search_term, all_titles, limit=10)
            
            if matches:
                st.write("Select a book:")
                
                # Display matches as buttons
                for i, (title, score) in enumerate(matches):
                    if st.button(f"{title} ({score}% match)", key=f"match_{i}"):
                        st.session_state.selected_book = title
                
                if 'selected_book' in st.session_state:
                    selected_book = st.session_state.selected_book
                    
                    # Number of recommendations
                    num_recs = st.slider("Number of recommendations", 3, 10, 5)
                    
                    # Get recommendations
                    with st.spinner("Finding similar books..."):
                        recommendations = get_recommends(
                            selected_book, 
                            title_to_isbn, 
                            isbn_to_title, 
                            book_matrix, 
                            model_knn, 
                            df_pivot,
                            n_neighbors=num_recs+1
                        )
                    
                    # Display results
                    st.subheader(f"Because you liked: *{recommendations[0]}*")
                    
                    for i, (book, score) in enumerate(recommendations[1]):
                        st.markdown(f"**{i+1}. {book}**")
                        # FIX: Convert numpy float32 to Python float
                        st.progress(float(min(1.0, score)))
                        st.caption(f"Similarity: {score:.2f}")
                        st.divider()
            else:
                st.warning("No matching books found. Try a different search term.")
        else:
            st.info("Start typing a book title above to get recommendations")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
