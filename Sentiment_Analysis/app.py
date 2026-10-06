import os
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

from src.predictor import SentimentPredictor
from src.preprocess import preprocess

# Set Streamlit Page Config
st.set_page_config(
    page_title="ReviewLens | Movie Review Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Responsive, Modern Cinema UI)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Gradient Headers */
    .main-title {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F68 50%, #FFAE33 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .sub-title {
        color: #8E9BAE;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphic Metric Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(255, 75, 75, 0.15);
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #9AA7B7;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Verdict Banner */
    .verdict-banner-pos {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.15) 0%, rgba(46, 213, 115, 0.05) 100%);
        border: 1px solid rgba(46, 213, 115, 0.35);
        color: #2ed573;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        margin: 1rem 0;
    }

    .verdict-banner-neg {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.15) 0%, rgba(255, 71, 87, 0.05) 100%);
        border: 1px solid rgba(255, 71, 87, 0.35);
        color: #ff4757;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        margin: 1rem 0;
    }

    .verdict-banner-mix {
        background: linear-gradient(135deg, rgba(255, 177, 66, 0.15) 0%, rgba(255, 177, 66, 0.05) 100%);
        border: 1px solid rgba(255, 177, 66, 0.35);
        color: #eccc68;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* Pills & Badges */
    .tag-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-pos {
        background: rgba(46, 213, 115, 0.2);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.4);
    }
    .badge-neg {
        background: rgba(255, 71, 87, 0.2);
        color: #ff4757;
        border: 1px solid rgba(255, 71, 87, 0.4);
    }
    .badge-project {
        background: rgba(255, 75, 75, 0.12);
        color: #FF6B6B;
        border: 1px solid rgba(255, 75, 75, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.6rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Cache Model and Artifacts
@st.cache_resource(show_spinner="Loading pickled ML pipeline artifacts...")
def get_predictor():
    return SentimentPredictor()

@st.cache_data(show_spinner="Loading 10,000 dataset...")
def load_main_dataset():
    paths = [
        "data/movie_reviews_10k.csv",
        "Sentiment_Analysis/data/movie_reviews_10k.csv",
        "Sentiment_Analysis/data/movie_reviews.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_model_metrics():
    paths = [
        "models/model_metrics.json",
        "Sentiment_Analysis/models/model_metrics.json"
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    return {}

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Failed to load pickled model artifacts: {e}")
    st.info("Please run `python train.py` to train and serialize the models first.")
    st.stop()

df_10k = load_main_dataset()
metrics_data = load_model_metrics()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:2.2rem;">🎬</span>
            <div>
                <h2 style="margin:0; font-size:1.4rem; font-weight:800; color:#FF4B4B;">ReviewLens AI</h2>
                <span style="font-size:0.8rem; color:#8E9BAE;">Movie Sentiment Analysis</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Active ML Algorithm")

    champion_name = metrics_data.get('champion_model', 'Linear SVM')
    algo_options = [
        f"🏆 Champion: {champion_name}",
        "Linear SVM",
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest"
    ]
    selected_algo_label = st.selectbox(
        "Choose Model for Predictions:",
        options=algo_options,
        index=0,
        help="Select which trained ML model to run for real-time predictions and batch analysis."
    )

    # Map label to model name
    if "Champion" in selected_algo_label:
        active_model_name = "Champion (Best Selected)"
    else:
        active_model_name = selected_algo_label

    predictor.set_model(active_model_name)

    st.markdown("---")
    st.markdown("### ⚙️ Engine Specifications")
    champ_acc = metrics_data.get('accuracy', 0.8980) * 100
    champ_r2 = metrics_data.get('r2_score', 0.6866)
    champ_f1 = metrics_data.get('f1_score', 0.8980)

    st.markdown(f"""
    - **Dataset**: 10,000 Balanced IMDB Reviews
    - **Active Model**: `{predictor.current_model_name}`
    - **Champion Model**: **{champion_name}**
    - **Top Accuracy**: **{champ_acc:.2f}%**
    - **Top R² Score**: **{champ_r2:.4f}**
    - **Top F1-Score**: **{champ_f1:.4f}**
    - **Serialization**: Python **Pickle (`.pkl`)**
    """)

    st.markdown("---")
    st.markdown("### 📥 Quick Sample Downloads")
    sample_path = "data/sample_movie_reviews.csv"
    if not os.path.exists(sample_path):
        sample_path = "Sentiment_Analysis/data/sample_movie_reviews.csv"

    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            sample_bytes = f.read()
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_bytes,
            file_name="sample_movie_reviews.csv",
            mime="text/csv",
            help="Download a ready-to-test CSV with 'movie_title', 'review', and 'sentiment' columns."
        )

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info(
        "• All 4 algorithms (SVM, Logistic Regression, Naive Bayes, Random Forest) are trained on the 10k dataset.\n"
        "• The system automatically picked the model with the highest Accuracy and R² score as the Champion!\n"
        "• Your uploaded CSV only needs a **`review`** column."
    )

# --- HEADER SECTION ---
st.markdown("""
    <div class="badge-project">🚀 ReviewLens AI • Production Streamlit Dashboard</div>
    <div class="main-title">Movie Review Sentiment Analysis Engine</div>
    <div class="sub-title">Multi-Algorithm NLP & Machine Learning Suite with Automated Model Selection & Audience Forecasting</div>
""", unsafe_allow_html=True)

# Top KPI Metric Strip
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
with col_kpi1:
    st.markdown("""
        <div class="kpi-card">
            <div class="kpi-lbl">Trained Dataset</div>
            <div class="kpi-val" style="color:#FF4B4B;">10,000</div>
        </div>
    """, unsafe_allow_html=True)
with col_kpi2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-lbl">Champion Accuracy</div>
            <div class="kpi-val" style="color:#2ed573;">{champ_acc:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
with col_kpi3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-lbl">Champion R² Score</div>
            <div class="kpi-val" style="color:#1e90ff;">{champ_r2:.3f}</div>
        </div>
    """, unsafe_allow_html=True)
with col_kpi4:
    st.markdown("""
        <div class="kpi-card">
            <div class="kpi-lbl">Algorithms Trained</div>
            <div class="kpi-val" style="color:#ffa502;">4 Models</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# --- APPLICATION TABS ---
tab_movie, tab_live, tab_csv, tab_eval, tab_guide = st.tabs([
    "🎬 Movie Audience Analysis",
    "⚡ Real-Time Review Tester",
    "📁 Custom CSV Batch Analyzer",
    "📊 Model Comparison & Diagnostics",
    "📖 CSV Format & User Guide"
])

# =========================================================================
# TAB 1: MOVIE AUDIENCE SENTIMENT ANALYSIS
# =========================================================================
with tab_movie:
    st.markdown("### 🎬 Predict Overall Audience Reception by Movie")
    st.markdown(
        f"Analyzing audience consensus using active model: **`{predictor.current_model_name}`**. "
        "Select a movie from the 10,000 dataset or type a title to see the aggregate audience consensus, "
        "positive/negative split, and confidence metrics."
    )

    if not df_10k.empty and 'movie_title' in df_10k.columns:
        movie_list = sorted(df_10k['movie_title'].dropna().unique().tolist())
    else:
        movie_list = ["Inception", "The Dark Knight", "Interstellar", "Titanic", "Avatar"]

    mcol1, mcol2 = st.columns([2, 1])
    with mcol1:
        selected_movie = st.selectbox(
            "Select a Movie Title from Dataset:",
            options=movie_list,
            index=0
        )
    with mcol2:
        custom_movie_search = st.text_input(
            "Or Type Custom Movie Name:",
            placeholder="e.g. Oppenheimer, Barbie..."
        )

    active_movie = custom_movie_search.strip() if custom_movie_search.strip() else selected_movie

    # Filter dataset for this movie
    movie_reviews_df = pd.DataFrame()
    if not df_10k.empty and 'movie_title' in df_10k.columns:
        movie_reviews_df = df_10k[df_10k['movie_title'].astype(str).str.contains(active_movie, case=False, na=False)].copy()

    if movie_reviews_df.empty:
        st.warning(f"No pre-indexed reviews found for **'{active_movie}'** in the 10k dataset. Try picking another movie from the dropdown or upload your own reviews in Tab 3!")
    else:
        st.markdown(f"#### 📊 Audience Consensus Dashboard: **{active_movie}**")
        
        with st.spinner(f"Analyzing {len(movie_reviews_df)} audience reviews for {active_movie}..."):
            pred_df, summary = predictor.predict_batch(movie_reviews_df, review_col='review', movie_name=active_movie)

        # Audience Verdict Banner
        status = summary['verdict_status']
        if status == 'positive':
            banner_class = "verdict-banner-pos"
        elif status == 'negative':
            banner_class = "verdict-banner-neg"
        else:
            banner_class = "verdict-banner-mix"

        st.markdown(f"""
            <div class="{banner_class}">
                🎯 Audience Verdict for <b>{active_movie}</b>: {summary['verdict']}<br>
                <span style="font-size:0.95rem; font-weight:500;">
                    {summary['positive_pct']}% Positive ({summary['positive_count']} reviews) vs {summary['negative_pct']}% Negative ({summary['negative_count']} reviews)
                </span>
            </div>
        """, unsafe_allow_html=True)

        # Audience Metrics Row
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Total Analyzed Reviews", summary['total_reviews'])
        with mc2:
            st.metric("Positive Reviews", f"{summary['positive_count']} ({summary['positive_pct']}%)")
        with mc3:
            st.metric("Negative Reviews", f"{summary['negative_count']} ({summary['negative_pct']}%)")
        with mc4:
            st.metric("Avg Prediction Confidence", f"{summary['avg_confidence']}%")

        # Interactive Visualizations
        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            # Donut Chart for Sentiment Breakdown
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative'],
                values=[summary['positive_count'], summary['negative_count']],
                hole=0.55,
                marker=dict(colors=['#2ed573', '#ff4757']),
                textinfo='percent+label',
                hoverinfo='label+value+percent'
            )])
            pie_fig.update_layout(
                title=f"Audience Sentiment Ratio ({active_movie})",
                margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#CBD5E0")
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        with vcol2:
            # Confidence Histogram
            conf_fig = px.histogram(
                pred_df,
                x='confidence_score',
                color='predicted_sentiment',
                color_discrete_map={'positive': '#2ed573', 'negative': '#ff4757'},
                nbins=20,
                title=f"Prediction Confidence Distribution (%)",
                labels={'confidence_score': 'Confidence Score (%)', 'count': 'Number of Reviews'}
            )
            conf_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#CBD5E0"),
                bargap=0.1
            )
            st.plotly_chart(conf_fig, use_container_width=True)

        # Reviews Inspection Table
        st.markdown(f"#### 🔍 Sample Audience Reviews for {active_movie}")
        show_sentiment_filter = st.radio(
            "Filter Reviews by Sentiment:",
            ["All", "Positive Only", "Negative Only"],
            horizontal=True
        )

        display_df = pred_df[['review', 'predicted_sentiment', 'confidence_score']].copy()
        if show_sentiment_filter == "Positive Only":
            display_df = display_df[display_df['predicted_sentiment'] == 'positive']
        elif show_sentiment_filter == "Negative Only":
            display_df = display_df[display_df['predicted_sentiment'] == 'negative']

        st.dataframe(
            display_df.head(20).rename(columns={
                'review': 'Raw Review Text',
                'predicted_sentiment': 'Predicted Sentiment',
                'confidence_score': 'Confidence (%)'
            }),
            use_container_width=True,
            height=320
        )

# =========================================================================
# TAB 2: REAL-TIME SINGLE & MULTI-REVIEW TESTER
# =========================================================================
with tab_live:
    st.markdown("### ⚡ Real-Time Sentiment Prediction")
    st.markdown(f"Running inference with active model: **`{predictor.current_model_name}`**")

    sample_buttons_col1, sample_buttons_col2, sample_buttons_col3 = st.columns(3)
    sample_text = ""
    with sample_buttons_col1:
        if st.button("🌟 Test Positive Review"):
            st.session_state['input_review'] = (
                "An absolute cinematic masterpiece! The cinematography was breathtaking, "
                "the acting was profoundly moving, and the screenplay kept me glued to my seat. A 10/10 triumph."
            )
    with sample_buttons_col2:
        if st.button("❌ Test Negative Review"):
            st.session_state['input_review'] = (
                "A complete disaster of a film. The plot was incoherent, the acting felt robotic, "
                "and two hours felt like an eternity. Definitely the worst movie I have seen all year."
            )
    with sample_buttons_col3:
        if st.button("⚖️ Test Nuanced Review"):
            st.session_state['input_review'] = (
                "The visual effects were undeniably gorgeous and the score was great, "
                "yet the pacing dragged painfully and the character motivations made no sense."
            )

    default_input = st.session_state.get('input_review', "Inception is one of the most innovative and thrilling sci-fi movies ever created. Truly brilliant direction!")
    review_input = st.text_area(
        "Enter Movie Review:",
        value=default_input,
        height=140,
        placeholder="Type or paste audience movie review here..."
    )

    if st.button("🚀 Analyze Sentiment", type="primary"):
        if review_input.strip():
            result = predictor.predict_single(review_input)
            
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                is_pos = (result['sentiment'] == 'positive')
                badge_color = "#2ed573" if is_pos else "#ff4757"
                icon = "🎉 POSITIVE" if is_pos else "🛑 NEGATIVE"

                st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:1.5rem; text-align:center;">
                        <span style="font-size:0.9rem; color:#8E9BAE; font-weight:600; text-transform:uppercase;">Classification Result ({result['model_used']})</span>
                        <h2 style="color:{badge_color}; margin:0.3rem 0; font-size:2.2rem; font-weight:800;">{icon}</h2>
                        <span style="font-size:1.15rem; font-weight:600;">Confidence: <b>{result['confidence']}%</b></span>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                st.markdown("#### 🔑 Influential Sentiment Tokens")
                if result['sentiment_words']:
                    tokens_html = ""
                    for word_obj in result['sentiment_words']:
                        pol = word_obj['polarity']
                        cls_name = "badge-pos" if pol == "positive" else "badge-neg"
                        sign = "+" if pol == "positive" else "-"
                        tokens_html += f'<span class="tag-badge {cls_name}">{word_obj["word"]} ({sign}{abs(word_obj["weight"])})</span>'
                    st.markdown(tokens_html, unsafe_allow_html=True)
                else:
                    st.caption("No specific high-weight indicator tokens extracted.")

            with res_col2:
                # Gauge Chart for Probability
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['positive_prob'],
                    title={'text': "Positive Sentiment Probability (%)", 'font': {'size': 16, 'color': '#CBD5E0'}},
                    number={'suffix': "%", 'font': {'color': '#2ed573' if is_pos else '#ff4757'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#718096"},
                        'bar': {'color': "#2ed573" if is_pos else "#ff4757"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(255, 71, 87, 0.15)"},
                            {'range': [50, 100], 'color': "rgba(46, 213, 115, 0.15)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                gauge_fig.update_layout(
                    height=240,
                    margin=dict(t=30, b=10, l=30, r=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#CBD5E0")
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            with st.expander("🛠️ View Cleaned Text (NLP Preprocessed)"):
                st.code(result['clean_text'] if result['clean_text'] else "(empty string after stopword & punctuation cleaning)", language="text")
        else:
            st.warning("Please type a review before running sentiment analysis.")

# =========================================================================
# TAB 3: CUSTOM CSV BATCH ANALYZER
# =========================================================================
with tab_csv:
    st.markdown("### 📁 Custom CSV Dataset Batch Analyzer")
    st.markdown(
        f"Upload any movie review CSV file to perform mass sentiment classification using **`{predictor.current_model_name}`**, "
        "compute audience statistics, and generate classification evaluation if true labels exist."
    )

    st.info(
        "💡 **CSV Requirements**:\n"
        "- **Required Column**: `review` (or `reviews`, `text`, `content`)\n"
        "- **Optional Columns**: `movie_title` (for filtering), `sentiment` (for automated confusion matrix & evaluation against true labels)"
    )

    csv_col1, csv_col2 = st.columns([2, 1])
    with csv_col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file to analyze:",
            type=["csv"],
            help="Upload your dataset file containing movie reviews."
        )
    with csv_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        use_sample_btn = st.button("🧪 Load Sample 25-Review CSV")

    target_df = None
    if uploaded_file is not None:
        try:
            target_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV with {len(target_df)} rows and columns: `{list(target_df.columns)}`")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
    elif use_sample_btn:
        sample_path = "data/sample_movie_reviews.csv"
        if not os.path.exists(sample_path):
            sample_path = "Sentiment_Analysis/data/sample_movie_reviews.csv"
        if os.path.exists(sample_path):
            target_df = pd.read_csv(sample_path)
            st.success(f"Loaded built-in sample test dataset with {len(target_df)} reviews.")

    if target_df is not None:
        candidate_cols = [c for c in target_df.columns if c.strip().lower() in ['review', 'reviews', 'text', 'comment', 'content']]
        selected_review_col = st.selectbox(
            "Select the column containing review text:",
            options=target_df.columns,
            index=list(target_df.columns).index(candidate_cols[0]) if candidate_cols else 0
        )

        has_movie_col = any(c.strip().lower() in ['movie_title', 'movie', 'title', 'film'] for c in target_df.columns)
        filter_movie_val = ""
        if has_movie_col:
            movie_col_name = [c for c in target_df.columns if c.strip().lower() in ['movie_title', 'movie', 'title', 'film']][0]
            unique_movies = ["All Movies"] + sorted(target_df[movie_col_name].dropna().unique().tolist())
            chosen_movie_filter = st.selectbox("Optional: Filter by Movie in this CSV:", unique_movies)
            if chosen_movie_filter != "All Movies":
                filter_movie_val = chosen_movie_filter

        if st.button("⚡ Run Full Batch Inference", type="primary"):
            with st.spinner(f"Processing reviews with {predictor.current_model_name}..."):
                processed_df, summary = predictor.predict_batch(
                    target_df,
                    review_col=selected_review_col,
                    movie_name=filter_movie_val if filter_movie_val else None
                )

            if processed_df.empty:
                st.warning("No records matched the selected criteria.")
            else:
                st.markdown(f"### 📊 Audience Sentiment Verdict: **{summary['movie_name']}**")
                
                status = summary['verdict_status']
                banner_cls = "verdict-banner-pos" if status == "positive" else ("verdict-banner-neg" if status == "negative" else "verdict-banner-mix")
                st.markdown(f"""
                    <div class="{banner_cls}">
                        <b>Verdict ({summary['model_used']}):</b> {summary['verdict']}<br>
                        <span style="font-size:0.95rem; font-weight:500;">
                            Positive: {summary['positive_pct']}% ({summary['positive_count']}) | Negative: {summary['negative_pct']}% ({summary['negative_count']})
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                with b_col1:
                    st.metric("Total Reviews Analyzed", summary['total_reviews'])
                with b_col2:
                    st.metric("Positive Count", f"{summary['positive_count']} ({summary['positive_pct']}%)")
                with b_col3:
                    st.metric("Negative Count", f"{summary['negative_count']} ({summary['negative_pct']}%)")
                with b_col4:
                    st.metric("Avg Prediction Confidence", f"{summary['avg_confidence']}%")

                b_fig_col1, b_fig_col2 = st.columns(2)
                with b_fig_col1:
                    chart_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative'],
                        'Count': [summary['positive_count'], summary['negative_count']]
                    })
                    bar_fig = px.bar(
                        chart_df,
                        x='Sentiment',
                        y='Count',
                        color='Sentiment',
                        color_discrete_map={'Positive': '#2ed573', 'Negative': '#ff4757'},
                        title="Sentiment Distribution",
                        text='Count'
                    )
                    bar_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
                    st.plotly_chart(bar_fig, use_container_width=True)

                with b_fig_col2:
                    donut_fig = go.Figure(data=[go.Pie(
                        labels=['Positive', 'Negative'],
                        values=[summary['positive_count'], summary['negative_count']],
                        hole=0.6,
                        marker=dict(colors=['#2ed573', '#ff4757'])
                    )])
                    donut_fig.update_layout(title="Audience Consensus Share", paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
                    st.plotly_chart(donut_fig, use_container_width=True)

                # Automated Evaluation / Confusion Matrix if ground truth present
                gt_col_candidates = [c for c in processed_df.columns if c.strip().lower() in ['sentiment', 'sentiment_true', 'true_sentiment', 'label']]
                if gt_col_candidates:
                    gt_col = gt_col_candidates[0]
                    st.markdown("---")
                    st.markdown(f"#### 🎯 Evaluation vs Ground Truth Column: `{gt_col}`")
                    
                    y_true = processed_df[gt_col].astype(str).str.lower().str.strip()
                    y_pred = processed_df['predicted_sentiment'].astype(str).str.lower().str.strip()
                    
                    valid_mask = y_true.isin(['positive', 'negative'])
                    if valid_mask.sum() > 0:
                        eval_cm = confusion_matrix(y_true[valid_mask], y_pred[valid_mask], labels=['positive', 'negative'])
                        
                        ev1, ev2 = st.columns([1, 1])
                        with ev1:
                            cm_fig = px.imshow(
                                eval_cm,
                                text_auto=True,
                                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                x=['Positive', 'Negative'],
                                y=['Positive', 'Negative'],
                                color_continuous_scale='Blues',
                                title=f"Batch Confusion Matrix ({summary['model_used']})"
                            )
                            cm_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#CBD5E0"))
                            st.plotly_chart(cm_fig, use_container_width=True)

                        with ev2:
                            eval_rep = classification_report(y_true[valid_mask], y_pred[valid_mask], output_dict=True)
                            rep_df = pd.DataFrame(eval_rep).transpose().round(3)
                            st.markdown("<b>Classification Metrics Report:</b>", unsafe_allow_html=True)
                            st.dataframe(rep_df, use_container_width=True)

                st.markdown("#### 📥 Download Enriched Prediction Dataset")
                export_cols = [c for c in processed_df.columns if c != 'clean_review']
                csv_data = processed_df[export_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Full Results (CSV)",
                    data=csv_data,
                    file_name="sentiment_analyzed_results.csv",
                    mime="text/csv",
                    type="primary"
                )

                st.dataframe(processed_df[export_cols].head(25), use_container_width=True)

# =========================================================================
# TAB 4: MODEL EVALUATION, COMPARISON & DIAGNOSTICS
# =========================================================================
with tab_eval:
    st.markdown("### 📊 Multi-Algorithm Benchmark & Model Comparison")
    st.markdown(
        "Trained on the **10,000 dataset** (8,000 train samples, 2,000 holdout test samples). "
        "The model with the highest **Accuracy and R² Score** was automatically selected as the **Champion Model**."
    )

    leaderboard_data = metrics_data.get('leaderboard', [])
    all_models_metrics = metrics_data.get('all_models_metrics', {})
    champ_model_name = metrics_data.get('champion_model', 'Linear SVM')

    if leaderboard_data:
        # Leaderboard Table
        lb_df = pd.DataFrame(leaderboard_data)
        lb_display = lb_df.rename(columns={
            'model_name': 'Machine Learning Model',
            'accuracy': 'Accuracy',
            'r2_score': 'R² Score',
            'f1_score': 'F1-Score',
            'precision': 'Precision',
            'recall': 'Recall'
        })
        # Add badge to champion
        lb_display['Machine Learning Model'] = lb_display['Machine Learning Model'].apply(
            lambda x: f"🏆 {x} (Champion)" if x == champ_model_name else x
        )
        lb_display['Accuracy'] = lb_display['Accuracy'].apply(lambda x: f"{x * 100:.2f}%")
        lb_display['R² Score'] = lb_display['R² Score'].apply(lambda x: f"{x:.4f}")
        lb_display['F1-Score'] = lb_display['F1-Score'].apply(lambda x: f"{x:.4f}")
        lb_display['Precision'] = lb_display['Precision'].apply(lambda x: f"{x:.4f}")
        lb_display['Recall'] = lb_display['Recall'].apply(lambda x: f"{x:.4f}")

        st.markdown("#### 🏆 Algorithms Leaderboard")
        st.dataframe(lb_display, use_container_width=True, hide_index=True)

        # Plotly Comparison Bar Chart
        comp_df = pd.DataFrame(leaderboard_data)
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Accuracy (%)',
            x=comp_df['model_name'],
            y=comp_df['accuracy'] * 100,
            marker_color='#2ed573',
            text=[f"{v*100:.2f}%" for v in comp_df['accuracy']],
            textposition='auto'
        ))
        fig_comp.add_trace(go.Bar(
            name='R² Score',
            x=comp_df['model_name'],
            y=comp_df['r2_score'],
            marker_color='#1e90ff',
            text=[f"{v:.3f}" for v in comp_df['r2_score']],
            textposition='auto'
        ))
        fig_comp.add_trace(go.Bar(
            name='F1-Score',
            x=comp_df['model_name'],
            y=comp_df['f1_score'],
            marker_color='#ffa502',
            text=[f"{v:.3f}" for v in comp_df['f1_score']],
            textposition='auto'
        ))
        fig_comp.update_layout(
            barmode='group',
            title="Algorithm Comparison Across Evaluation Metrics",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#CBD5E0"),
            yaxis=dict(title="Metric Value"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Deep Diagnostics: Confusion Matrix & Classification Report per Model")

        model_inspect = st.selectbox(
            "Select Model to Inspect Details:",
            options=list(all_models_metrics.keys()),
            index=list(all_models_metrics.keys()).index(champ_model_name) if champ_model_name in all_models_metrics else 0
        )

        inspected_data = all_models_metrics.get(model_inspect, {})
        inspected_cm = inspected_data.get('confusion_matrix', {}).get('matrix', [[0, 0], [0, 0]])
        inspected_rep = inspected_data.get('classification_report', {})

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            cm_fig = go.Figure(data=go.Heatmap(
                z=inspected_cm,
                x=['Positive (Pred)', 'Negative (Pred)'],
                y=['Positive (True)', 'Negative (True)'],
                colorscale='Teal',
                text=[[f"{inspected_cm[0][0]}<br>(True Pos)", f"{inspected_cm[0][1]}<br>(False Neg)"],
                      [f"{inspected_cm[1][0]}<br>(False Pos)", f"{inspected_cm[1][1]}<br>(True Neg)"]],
                texttemplate="%{text}",
                textfont={"size": 15, "family": "Plus Jakarta Sans"},
                showscale=True
            ))
            cm_fig.update_layout(
                title=f"Confusion Matrix: {model_inspect}",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#CBD5E0"),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        with col_d2:
            st.markdown(f"<b>Classification Report: {model_inspect}</b>", unsafe_allow_html=True)
            if inspected_rep:
                clean_rep = {}
                for k, v in inspected_rep.items():
                    if isinstance(v, dict):
                        clean_rep[k] = {
                            "Precision": round(v.get('precision', 0), 3),
                            "Recall": round(v.get('recall', 0), 3),
                            "F1-Score": round(v.get('f1-score', 0), 3),
                            "Support": int(v.get('support', 0))
                        }
                st.dataframe(pd.DataFrame(clean_rep).transpose(), use_container_width=True, height=230)

        st.markdown("---")
        st.markdown("#### 🔤 Influential Indicator Features (TF-IDF Weights)")
        fcol1, fcol2 = st.columns(2)

        with fcol1:
            top_pos = metrics_data.get('top_positive_features', [])[:15]
            if top_pos:
                pos_df = pd.DataFrame(top_pos)
                fig_pos = px.bar(
                    pos_df,
                    x='weight',
                    y='word',
                    orientation='h',
                    color_discrete_sequence=['#2ed573'],
                    title="Top Positive Predictor Words"
                )
                fig_pos.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#CBD5E0"),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_pos, use_container_width=True)

        with fcol2:
            top_neg = metrics_data.get('top_negative_features', [])[:15]
            if top_neg:
                neg_df = pd.DataFrame(top_neg)
                neg_df['magnitude'] = neg_df['weight'].abs()
                fig_neg = px.bar(
                    neg_df,
                    x='magnitude',
                    y='word',
                    orientation='h',
                    color_discrete_sequence=['#ff4757'],
                    title="Top Negative Predictor Words (Magnitude)"
                )
                fig_neg.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#CBD5E0"),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_neg, use_container_width=True)

# =========================================================================
# TAB 5: CSV FORMAT & USER GUIDE
# =========================================================================
with tab_guide:
    st.markdown("### 📖 Dataset Description & CSV Format Specification")
    st.markdown("""
        ReviewLens accepts CSV datasets for batch audience sentiment forecasting. 
        Follow the specifications below to ensure seamless processing:
    """)

    st.markdown("""
    #### 1. Supported CSV Column Structure
    | Column Name | Required | Data Type | Description |
    | :--- | :---: | :--- | :--- |
    | **`review`** | **Yes** | String (Text) | Raw movie review text (e.g. *"Great acting and plot!"*). Aliases accepted: `reviews`, `text`, `content`. |
    | **`movie_title`** | Optional | String | Name of the movie (e.g. *"Inception"*, *"Titanic"*). Used for movie filtering and audience forecasting. |
    | **`sentiment`** | Optional | String | Ground truth polarity (`positive` or `negative`). If supplied, automated confusion matrix and F1 scores are calculated! |
    """)

    st.markdown("#### 2. Example CSV File Content")
    st.code("""movie_title,review,sentiment
Inception,"A visual and intellectual masterpiece that keeps you engaged.",positive
Titanic,"Heartbreaking story with unforgettable performances and soundtrack.",positive
The Room,"Horrible acting, nonsensical dialogue, and bizarre editing.",negative
Gladiator,"Russell Crowe delivers an iconic performance with epic battle scenes.",positive""", language="csv")

    st.markdown("#### 3. Algorithms & ML Pipeline Details")
    st.markdown("""
    The system benchmarks 4 classical Machine Learning algorithms on TF-IDF word vectors:
    1. **Linear SVM (Calibrated)**: High-dimensional maximum margin classifier with calibrated probability output (**Champion Model**).
    2. **Logistic Regression**: High-efficiency linear probability classifier with strong generalization.
    3. **Multinomial Naive Bayes**: Probabilistic word frequency classifier fast for textual distributions.
    4. **Random Forest**: Non-linear ensemble model combining decision trees.

    All models are serialized as standard Python **Pickle (`.pkl`)** binaries for sub-millisecond production inference.
    """)

# --- FOOTER ---
st.markdown("""
    <div class="footer-text">
        ReviewLens • Movie Review Sentiment Analysis System • Production Streamlit Edition
    </div>
""", unsafe_allow_html=True)
