# --- 1. Importations ---
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import PunktSentenceTokenizer

# --- 2. Téléchargement des ressources NLTK ---
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        return True
    except:
        return False

# Configuration de la page
st.set_page_config(
    page_title="Klamna - كلامنا", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .result-positive {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-neutral {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
        margin: 1rem 0;
    }
    
    .complex-analysis {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b35;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9ff;
        border: 2px solid #667eea;
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #f8f9ff;
        border: 2px solid #667eea;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Forcer l'utilisation du CPU
device = torch.device("cpu")

# --- 3. Dictionnaire des labels ---
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

# --- 4. Fonction pour détecter le script dominant ---
def detect_script(text):
    arabic_pattern = re.compile("[\u0600-\u06FF]")  # Arabe
    marbet_pattern = re.compile(r"\b(dir|matdirch|wach|ghadi|bghit|safi|hadi|haka|daba|bezaf)\b", re.IGNORECASE)  # Mots caractéristiques
    
    if arabic_pattern.search(text):
        return "arabic"
    elif marbet_pattern.search(text):
        return "marbet"
    else:
        return "latin"

# --- 5. Charger les modèles avec gestion d'erreur améliorée ---
@st.cache_resource
def load_arabic_model():
    try:
        # Chemins possibles (local ou Google Drive)
        paths = [
            "C:/Users/HP/Downloads/darija_bert_model",
            "/content/drive/MyDrive/darija_bert_model"
        ]
        
        for path in paths:
            if os.path.exists(path):
                model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                return model, tokenizer, True
        
        raise FileNotFoundError("Modèle arabe non trouvé")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle arabe: {e}")
        return None, None, False

@st.cache_resource
def load_latin_model():
    try:
        # Chemins possibles (local ou Google Drive)
        paths = [
            "C:/Users/HP/Downloads/darijabert_arabizi_finetuned",
            "/content/drive/MyDrive/darijabert_arabizi_finetuned"
        ]
        
        for path in paths:
            if os.path.exists(path):
                model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                return model, tokenizer, True
                
        raise FileNotFoundError("Modèle latin non trouvé")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle latin: {e}")
        return None, None, False

@st.cache_resource
def load_marbet_model():
    try:
        path = "C:/Users/HP/Downloads/marbetv2_finetuned"
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return model, tokenizer, True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle Marbet: {e}")
        return None, None, False

# --- 6. Prédiction du sentiment automatique ---
def predict_sentiment_auto(comment):
    script = detect_script(comment)
    
    if script == "arabic":
        model, tokenizer, success = load_arabic_model()
        model_name = "DarijaBERT (Arabe)"
    elif script == "latin":
        model, tokenizer, success = load_latin_model()
        model_name = "DarijaBERT (Arabizi)"
    else:
        model, tokenizer, success = load_marbet_model()
        model_name = "Marbet v2"
    
    if not success or model is None:
        return None, None, script, model_name
    
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
    
    predicted_class_id = torch.argmax(logits).item()
    predicted_label = id2label[predicted_class_id]
    scores = {id2label[i]: round(probs[i], 4) for i in range(len(probs))}
    
    return predicted_label, scores, script, model_name

# --- 7. Analyse de commentaires complexes ---
def analyze_complex_comment(comment):
    """Analyse les commentaires complexes en les séparant par conjonctions"""
    # Séparation selon les conjonctions usuelles (darija, français, arabe)
    parts = re.split(r'\b(?:walakin|mais|ولكن|w|o|but|however|ma)\b', comment, flags=re.IGNORECASE)
    
    results = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            label, scores, script, model_name = predict_sentiment_auto(part)
            if label is not None:
                results.append({
                    'partie': i + 1,
                    'texte': part,
                    'label': label,
                    'score': scores[label],
                    'script': script,
                    'model': model_name,
                    'all_scores': scores
                })
    
    return results

# Prédire un seul commentaire avec choix de modèle (fonction originale maintenue)
def predict_sentiment(text, model_choice="auto"):
    if model_choice == "auto":
        return predict_sentiment_auto(text)
    else:
        script = detect_script(text)
        if model_choice == "arabic":
            model, tokenizer, success = load_arabic_model()
            model_name = "DarijaBERT (Arabe)"
        elif model_choice == "latin":
            model, tokenizer, success = load_latin_model()
            model_name = "DarijaBERT (Arabizi)"
        else:
            model, tokenizer, success = load_marbet_model()
            model_name = "Marbet v2"
        
        if not success or model is None:
            return None, None, script, model_name
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        labels = ["Negative", "Neutral", "Positive"]
        predicted_label = labels[probs.index(max(probs))]
        
        return predicted_label, probs, script, model_name

# Prédire un fichier CSV
def predict_batch(df, model_choice="auto"):
    results = []
    progress_bar = st.progress(0)
    total_comments = len(df)
    
    for i, comment in enumerate(df["commentaire"]):
        if model_choice == "auto":
            label, scores, script, model_name = predict_sentiment_auto(comment)
            if label is not None:
                results.append({
                    "Commentaire": comment,
                    "Script détecté": script,
                    "Modèle utilisé": model_name,
                    "Label": label,
                    "Confiance": scores[label],
                    "Score Négatif": scores.get("Negative", 0),
                    "Score Neutre": scores.get("Neutral", 0),
                    "Score Positif": scores.get("Positive", 0)
                })
        else:
            label, probs, script, model_name = predict_sentiment(comment, model_choice)
            if label is not None:
                results.append({
                    "Commentaire": comment,
                    "Script détecté": script,
                    "Modèle utilisé": model_name,
                    "Label": label,
                    "Confiance": max(probs) if probs else 0,
                    "Score Négatif": probs[0] if probs else 0,
                    "Score Neutre": probs[1] if probs else 0,
                    "Score Positif": probs[2] if probs else 0
                })
        progress_bar.progress((i + 1) / total_comments)
    
    return pd.DataFrame(results)

# Initialisation des ressources NLTK
if download_nltk_resources():
    st.success("📚 Ressources NLTK chargées avec succès")
else:
    st.warning("⚠️ Impossible de charger les ressources NLTK")

# Header principal
st.markdown("""
<div class="main-header">
    <h1>Klamna - كلامنا</h1>
    <h3>Plateforme d'analyse de sentiment pour le Darija marocain</h3>
    <p>Analysez vos textes en darija avec nos modèles IA avancés</p>
</div>
""", unsafe_allow_html=True)

# Sidebar moderne
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">📚 Navigation</h2>
</div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "",
    ["Nos modèles", "💬 Analyse simple", "🔍 Analyse complexe", "📁 Analyse par fichier", "📊 Statistiques"],
    index=1
)

# Section 1 : Nos modèles
if section == "Nos modèles":
    st.header(" Modèles d'IA disponibles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>🔤 DarijaBERT Arabe</h3>
            <p><strong>Spécialité:</strong> Textes en écriture arabe</p>
            <p><strong>Précision:</strong> 74.63%</p>
            <p><strong>Paramètres:</strong> ~147M</p>
            <p><strong>Usage:</strong> Darija écrit en caractères arabes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>🔤 DarijaBERT Arabizi</h3>
            <p><strong>Spécialité:</strong> Textes en écriture latine</p>
            <p><strong>Précision:</strong> 83.32%</p>
            <p><strong>Paramètres:</strong> ~170M</p>
            <p><strong>Usage:</strong> Darija écrit en lettres latines</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <h3>🔤 Marbet v2</h3>
            <p><strong>Spécialité:</strong> Modèle personnalisé</p>
            <p><strong>Précision:</strong> 66%</p>
            <p><strong>Type:</strong> Fine-tuned BERT</p>
            <p><strong>Usage:</strong> Darija mixte et spécialisé</p>
        </div>
        """, unsafe_allow_html=True)

# Section 2 : Analyse simple
elif section == "💬 Analyse simple":
    st.header("💬 Analyseur de sentiment intelligent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "✍️ Entrez votre texte en Darija:",
            height=150,
            placeholder="مثال: هاد الفيلم زوين بزاف\nExemple: had l film zwin bezaf\nExemple: Ce film est très beau"
        )
    
    with col2:
        st.markdown("### ⚙️ Paramètres")
        model_choice = st.selectbox(
            "Choisir le modèle:",
            ["auto", "arabic", "latin", "marbet"],
            format_func=lambda x: {
                "auto": "🤖 Détection automatique",
                "arabic": "🔤 DarijaBERT Arabe",
                "latin": "🔤 DarijaBERT Arabizi", 
                "marbet": "🔤 Marbet v2"
            }[x]
        )
        
        analyze_btn = st.button(" Analyser", type="primary", use_container_width=True)
    
    if analyze_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Veuillez entrer un texte à analyser.")
        else:
            with st.spinner("🔄 Analyse en cours..."):
                if model_choice == "auto":
                    label, scores, script, model_name = predict_sentiment_auto(user_input)
                    if label is not None:
                        confidence = scores[label]
                        score_list = [scores["Negative"], scores["Neutral"], scores["Positive"]]
                    else:
                        confidence = 0
                        score_list = [0, 0, 0]
                else:
                    label, score_list, script, model_name = predict_sentiment(user_input, model_choice)
                    confidence = max(score_list) if score_list else 0
                
                if label is not None:
                    # Affichage des résultats avec design moderne
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if label == "Positive":
                            st.markdown(f"""
                            <div class="result-positive">
                                <h3>😊 Sentiment: {label}</h3>
                                <p>Confiance: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif label == "Negative":
                            st.markdown(f"""
                            <div class="result-negative">
                                <h3>😔 Sentiment: {label}</h3>
                                <p>Confiance: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-neutral">
                                <h3>😐 Sentiment: {label}</h3>
                                <p>Confiance: {confidence*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🔍 Script détecté</h4>
                            <h3>{script.title()}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🤖 Modèle utilisé</h4>
                            <h3>{model_name}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Graphique des scores
                    st.markdown("### 📊 Distribution des scores")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Négatif', 'Neutre', 'Positif'],
                            y=[score_list[0]*100, score_list[1]*100, score_list[2]*100],
                            marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f']
                        )
                    ])
                    fig.update_layout(
                        title="Scores de confiance par sentiment",
                        yaxis_title="Pourcentage (%)",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Erreur lors de l'analyse. Vérifiez que les modèles sont disponibles.")

# Section 3 : Analyse complexe (NOUVELLE)
elif section == "🔍 Analyse complexe":
    st.header("🔍 Analyse de commentaires complexes")
    
    st.markdown("""
    <div class="model-card">
        <h4>Fonctionnalité avancée:</h4>
        <p>Cette analyse peut traiter des commentaires contenant plusieurs sentiments séparés par des conjonctions comme:</p>
        <ul>
            <li><strong>Darija:</strong> walakin, w, o</li>
            <li><strong>Français:</strong> mais, however</li>
            <li><strong>Arabe:</strong> ولكن</li>
        </ul>
        <p><strong>Exemple:</strong> "الخدمة زوينة بزاف mais le prix est trop élevé"</p>
    </div>
    """, unsafe_allow_html=True)
    
    complex_input = st.text_area(
        "✍️ Entrez votre commentaire complexe:",
        height=150,
        placeholder="Exemple: الخدمة زوينة بزاف mais le prix est trop élevé\nExemple: had restaurant zwin walakin ghali bezaf"
    )
    
    if st.button("🔍 Analyser le commentaire complexe", type="primary"):
        if complex_input.strip() == "":
            st.warning("⚠️ Veuillez entrer un commentaire à analyser.")
        else:
            with st.spinner("🔄 Analyse complexe en cours..."):
                results = analyze_complex_comment(complex_input)
                
                if results:
                    st.markdown("### 📊 Résultats de l'analyse complexe")
                    
                    for result in results:
                        st.markdown(f"""
                        <div class="complex-analysis">
                            <h4>📝 Partie {result['partie']}: {result['label'].upper()}</h4>
                            <p><strong>Texte:</strong> {result['texte']}</p>
                            <p><strong>Confiance:</strong> {result['score']*100:.1f}% | <strong>Script:</strong> {result['script']} | <strong>Modèle:</strong> {result['model']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Graphique de comparaison des parties
                    if len(results) > 1:
                        st.markdown("### 📈 Comparaison des sentiments par partie")
                        
                        partie_labels = [f"Partie {r['partie']}" for r in results]
                        sentiment_labels = [r['label'] for r in results]
                        confidences = [r['score'] for r in results]
                        
                        colors = ['#ff6b6b' if s == 'Negative' else '#6bcf7f' if s == 'Positive' else '#ffd93d' for s in sentiment_labels]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=partie_labels,
                                y=[c*100 for c in confidences],
                                marker_color=colors,
                                text=sentiment_labels,
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title="Sentiment et confiance par partie",
                            yaxis_title="Confiance (%)",
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Résumé global
                    sentiment_counts = {}
                    for result in results:
                        sentiment = result['label']
                        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                    
                    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Résumé global</h4>
                        <p><strong>Sentiment dominant:</strong> {dominant_sentiment}</p>
                        <p><strong>Nombre de parties:</strong> {len(results)}</p>
                        <p><strong>Distribution:</strong> {dict(sentiment_counts)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Impossible d'analyser ce commentaire. Vérifiez que les modèles sont disponibles.")

# Section 4 : Analyse par fichier
elif section == "📁 Analyse par fichier":
    st.header("📁 Analyse de fichiers CSV")
    
    st.markdown("""
    <div class="model-card">
        <h4>📋 Instructions:</h4>
        <ul>
            <li>Votre fichier CSV doit contenir une colonne nommée <code>commentaire</code></li>
            <li>Chaque ligne représente un texte à analyser</li>
            <li>Formats supportés: CSV avec encodage UTF-8</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Téléchargez votre fichier CSV",
            type="csv",
            help="Glissez-déposez votre fichier ou cliquez pour parcourir"
        )
    
    with col2:
        st.markdown("### ⚙️ Paramètres d'analyse")
        batch_model_choice = st.selectbox(
            " Modèle pour l'analyse:",
            ["auto", "arabic", "latin", "marbet"],
            format_func=lambda x: {
                "auto": "🤖 Détection automatique",
                "arabic": "🔤 DarijaBERT Arabe",
                "latin": "🔤 DarijaBERT Arabizi",
                "marbet": "🔤 Marbet v2"
            }[x],
            key="batch_model"
        )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if "commentaire" not in df.columns:
                st.error("❌ Le fichier doit contenir une colonne nommée `commentaire`.")
            else:
                st.success(f"✅ Fichier chargé: {len(df)} commentaires détectés")
                
                # Aperçu du fichier
                st.markdown("### 👀 Aperçu des données")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button(" Lancer l'analyse complète", type="primary"):
                    with st.spinner("🔄 Analyse en cours... Cela peut prendre quelques minutes."):
                        result_df = predict_batch(df, batch_model_choice)
                        
                        st.success("✅ Analyse terminée avec succès!")
                        
                        # Affichage des résultats
                        st.markdown("### 📊 Résultats de l'analyse")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Bouton de téléchargement
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Télécharger les résultats (CSV)",
                            data=csv,
                            file_name=f"analyse_sentiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv',
                            type="primary"
                        )
                        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier: {e}")

# Section 5 : Statistiques
elif section == "📊 Statistiques":
    st.header("📊 Statistiques de performance")
    
    # Métriques des modèles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>74.63%</h3>
            <p>DarijaBERT Arabe</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>83.32%</h3>
            <p>DarijaBERT Arabizi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>66.00%</h3>
            <p>Marbet v2</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique comparatif
    st.markdown("### 📈 Comparaison des performances")
    
    models = ['DarijaBERT Arabe', 'DarijaBERT Arabizi', 'Marbet v2']
    accuracies = [74.63, 83.32, 66.00]
    parameters = [147, 170, 147]  # en millions
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Précision (%)',
        x=models,
        y=accuracies,
        marker_color='#667eea'
    ))
    
    fig.update_layout(
        title="Performance des modèles",
        yaxis_title="Précision (%)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Informations techniques
    st.markdown("### 🔧 Informations techniques")
    
    tech_data = {
        'Modèle': models,
        'Précision (%)': accuracies,
        'Paramètres (M)': parameters,
        'Type de texte': ['Arabe', 'Latin', 'Mixte'],
        'Taille sur disque': ['~590 MB', '~680 MB', '~590 MB']
    }
    
    tech_df = pd.DataFrame(tech_data)
    st.dataframe(tech_df, use_container_width=True)
    
    # Nouvelles métriques pour l'analyse complexe
    st.markdown("### 🔍 Fonctionnalités avancées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>🤖 Détection automatique de script</h4>
            <p>✅ Reconnaissance automatique du type d'écriture</p>
            <p>✅ Support de l'arabe, latin et darija mixte</p>
            <p>✅ Sélection automatique du meilleur modèle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>🔍 Analyse de commentaires complexes</h4>
            <p>✅ Séparation par conjonctions (walakin, mais, ولكن)</p>
            <p>✅ Analyse sentiment par partie</p>
            <p>✅ Résumé global avec sentiment dominant</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p> <strong>Klamna - كلامنا</strong> | Plateforme d'analyse de sentiment pour le Darija marocain</p>
    <p>Développé avec ❤️ en utilisant Streamlit et les modèles Transformers</p>
    <p><strong>Nouvelles fonctionnalités:</strong> Analyse automatique • Commentaires complexes • Support Google Colab</p>
</div>
""", unsafe_allow_html=True)