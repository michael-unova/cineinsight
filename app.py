import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from sentiment_model import predict_sentiment
from recommender import load_movie_data, build_recommender, get_recommendations, search_movies

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineInsight",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load models (cached so they only load once) ───────────────────────────────
@st.cache_resource
def load_sentiment_model():
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return model, vec

@st.cache_resource
def load_recommender_data():
    movies = load_movie_data()
    cosine_sim, indices = build_recommender(movies)
    return movies, cosine_sim, indices

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 CineInsight")
st.sidebar.markdown("AI-powered movie analysis & recommendations")
page = st.sidebar.radio("Navigate", ["Sentiment Analyzer", "Movie Recommender", "Data Explorer"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Sentiment Analyzer
# ─────────────────────────────────────────────────────────────────────────────
if page == "Sentiment Analyzer":
    st.title("Movie Review Sentiment Analyzer")
    st.markdown("Paste any movie review and the AI will tell you if it's positive or negative.")
    
    model, vec = load_sentiment_model()
    
    review_text = st.text_area(
        "Enter a movie review:",
        placeholder="e.g. This film was an absolute masterpiece with stunning visuals...",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_btn and review_text.strip():
        with st.spinner("Analyzing..."):
            label, confidence = predict_sentiment(review_text, model, vec)
        
        # Result display
        color = "green" if label == "Positive" else "red"
        icon  = "😊" if label == "Positive" else "😞"
        
        st.markdown(f"""
        <div style='padding:20px; border-radius:12px; border: 1px solid {color}; margin-top:16px'>
            <h2 style='color:{color}; margin:0'>{icon} {label}</h2>
            <p style='margin:4px 0 0'>Confidence: <strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Confidence %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if label == "Positive" else "red"},
                'steps': [
                    {'range': [0, 50],  'color': "#f8d7da"},
                    {'range': [50, 75], 'color': "#fff3cd"},
                    {'range': [75, 100],'color': "#d4edda"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    elif analyze_btn:
        st.warning("Please enter a review first.")
    
    # Batch analysis section
    st.divider()
    st.subheader("Batch Analysis (try multiple reviews)")
    sample_reviews = [
        "Absolutely breathtaking cinematography and a powerful story.",
        "Dull, predictable, and a complete waste of two hours.",
        "The acting was superb but the plot felt rushed.",
        "One of the worst films I've ever seen. Avoid at all costs.",
        "A magical experience that I'll remember for years."
    ]
    
    results = []
    for rev in sample_reviews:
        lbl, conf = predict_sentiment(rev, model, vec)
        results.append({"Review": rev[:70]+"...", "Sentiment": lbl, "Confidence": f"{conf:.1%}"})
    
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Movie Recommender
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Movie Recommender":
    st.title("Movie Recommendation Engine")
    st.markdown("Search for a movie and get AI-powered similar movie recommendations.")
    
    movies, cosine_sim, indices = load_recommender_data()
    
    search_query = st.text_input("Search for a movie:", placeholder="e.g. Toy Story, Matrix, Inception...")
    
    if search_query:
        search_results = search_movies(search_query, movies)
        
        if search_results.empty:
            st.warning("No movies found. Try a different title.")
        else:
            st.subheader(f"Search results for '{search_query}'")
            
            # Show top result prominently
            top = search_results.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Title", top['clean_title'])
            col2.metric("Avg Rating", f"{top['avg_rating']:.1f} / 5.0")
            col3.metric("Total Ratings", f"{top['num_ratings']:,}")
            
            st.markdown(f"**Genres:** {top['genres'].replace('|', ' · ')}")
            
            # Recommendations
            st.subheader("Recommended movies you might also like")
            recs = get_recommendations(search_query, movies, cosine_sim, indices)
            
            if not recs.empty:
                cols = st.columns(3)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div style='padding:12px; border:1px solid #ddd; border-radius:8px; margin-bottom:10px'>
                            <strong>{row['clean_title']}</strong><br>
                            <small>{row['genres'].replace('|',' · ')}</small><br>
                            ⭐ {row['avg_rating']:.1f} &nbsp;|&nbsp; {int(row['num_ratings']):,} ratings
                        </div>
                        """, unsafe_allow_html=True)
            
            # Full search results table
            with st.expander("See all search results"):
                display = search_results[['clean_title','genres','avg_rating','num_ratings','year']].copy()
                display.columns = ['Title','Genres','Avg Rating','# Ratings','Year']
                st.dataframe(display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Explorer":
    st.title("Movie Data Explorer")
    st.markdown("Explore the MovieLens dataset with interactive charts.")
    
    movies, _, _ = load_recommender_data()
    
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(movies):,}")
    col2.metric("Avg Rating", f"{movies['avg_rating'].mean():.2f}")
    col3.metric("Most Rated Genre", "Drama")
    col4.metric("Movies Rated >4.0", f"{(movies['avg_rating']>4).sum():,}")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Genre distribution")
        genres_exploded = movies['genres'].str.split('|').explode()
        genre_counts = genres_exploded.value_counts().head(15).reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig = px.bar(genre_counts, x='Count', y='Genre', orientation='h',
                     color='Count', color_continuous_scale='Teal')
        fig.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Rating distribution")
        rated = movies[movies['num_ratings'] > 10]
        fig2 = px.histogram(rated, x='avg_rating', nbins=30,
                            color_discrete_sequence=['#7F77DD'])
        fig2.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Top 20 highest-rated movies (min 100 ratings)")
    top_movies = movies[movies['num_ratings'] >= 100].nlargest(20, 'avg_rating')
    fig3 = px.bar(top_movies, x='avg_rating', y='clean_title',
                  orientation='h', color='avg_rating',
                  color_continuous_scale='Viridis',
                  labels={'avg_rating':'Avg Rating', 'clean_title':'Movie'})
    fig3.update_layout(height=500, margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Movies released per decade")
    decade_data = movies.dropna(subset=['year']).copy()
    decade_data['decade'] = (decade_data['year'] // 10 * 10).astype(int).astype(str) + 's'
    decade_counts = decade_data['decade'].value_counts().sort_index().reset_index()
    decade_counts.columns = ['Decade', 'Count']
    fig4 = px.bar(decade_counts, x='Decade', y='Count', color='Count',
                  color_continuous_scale='Blues')
    fig4.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig4, use_container_width=True)