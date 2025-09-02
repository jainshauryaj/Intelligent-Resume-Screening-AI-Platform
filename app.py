#!/usr/bin/env python3
"""
app.py
End-to-end Streamlit app for Smart Hiring Resume Screening with optional model training and NER training.
Run with: python3 -m streamlit run app.py
"""
import os
import glob
import json
import pickle
import tempfile
import logging
import ast
import re
import random

import pandas as pd
import numpy as np
import pdfplumber
import spacy
import nltk
import streamlit as st
import torch

from spacy.tokens import DocBin
import spacy.util

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from sentence_transformers import SentenceTransformer
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline as hf_pipeline
)
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
torch.cuda.empty_cache()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -- Utility functions --

def extract_text_from_pdf(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as doc:
        for page in doc.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n".join(texts)


def create_training_data(annot_path: str) -> DocBin:
    db = DocBin()
    nlp_blank = spacy.blank('en')
    with open(annot_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            doc = nlp_blank.make_doc(example['text'])
            ents = []
            for start, end, label in example['entities']:
                span = doc.char_span(start, end, label=label)
                if span:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)
    return db


def train_custom_ner(train_path: str, dev_path: str, output_dir: str, n_iter: int = 20):
    train_db = DocBin().from_disk(train_path) if train_path.endswith('.spacy') else create_training_data(train_path)
    dev_db   = DocBin().from_disk(dev_path)   if dev_path.endswith('.spacy')   else create_training_data(dev_path)

    nlp_custom = spacy.blank('en')
    ner = nlp_custom.add_pipe('ner')
    ner.add_label('SKILL')

    optimizer = nlp_custom.begin_training()
    docs = list(train_db.get_docs(nlp_custom.vocab))
    for itn in range(n_iter):
        random.shuffle(docs)
        losses = {}
        for batch in spacy.util.minibatch(docs, size=8):
            nlp_custom.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        logger.info(f"NER Iter {itn+1}/{n_iter}, losses={losses}")
    nlp_custom.to_disk(output_dir)
    logger.info(f"Saved custom NER model to {output_dir}")

# -- Caching heavy resources --
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('custom_ner_model')
    except:
        try:
            return spacy.load('en_core_web_trf')
        except:
            return spacy.load('en_core_web_sm')

@st.cache_resource
def load_hf_ner():
    try:
        return hf_pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')
    except:
        return None

@st.cache_resource
def load_baseline_model():
    data = pickle.load(open('baseline_lr_tuned.pkl','rb'))
    return data['model'], data['encoder']

@st.cache_resource
def load_transformer_resources():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model     = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    encoder   = pickle.load(open('best_transformer/label_encoder.pkl','rb'))
    return tokenizer, model, encoder

@st.cache_data
def load_job_descriptions(csv_path: str = 'job_descriptions.csv') -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_data
def load_ground_truth(csv_path: str = 'gt_skills.csv') -> pd.DataFrame:
    return pd.read_csv(csv_path)

# Initialize NLP & embeddings
en_model    = load_spacy_model()
hf_ner      = load_hf_ner()
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()
stemmer     = PorterStemmer()
sbert_model = load_sbert_model()

# === Cell 3: Skill Extraction via NER ===
try:
    with open('auto_skill_dict.json', 'r') as f:
        SKILL_SYNONYMS = pd.read_json(f, typ='series').to_dict()
    logger.info('Loaded skill synonyms from auto_skill_dict.json')
except Exception:
    SKILL_SYNONYMS = {}
    logger.warning('auto_skill_dict.json not found; using empty mapping')


def extract_skills(text: str) -> list:
    doc = en_model(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ('SKILL', 'ORG', 'PRODUCT'):
            skill = ent.text.lower().strip()
            skill = SKILL_SYNONYMS.get(skill, skill)
            skills.add(skill)
    return list(skills)

# -- Preprocessing & embeddings --
@st.cache_data
def preprocess_text(text: str) -> str:
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens)

@st.cache_data
def generate_embeddings(texts):
    return sbert_model.encode(texts, show_progress_bar=False)

@st.cache_data
def load_dataset(data_dir: str):
    texts, labels, skills_list = [], [], []
    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder): continue
        for pdf_file in glob.glob(os.path.join(folder, '*.pdf')):
            raw  = extract_text_from_pdf(pdf_file)
            proc = preprocess_text(raw)
            texts.append(proc)
            labels.append(label)
            skills_list.append(extract_skills(raw))
    return texts, labels, skills_list

# Dataset for transformer
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.enc    = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Shared metrics for transformers
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='micro', zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# Model training functions
def train_classifier(data_dir: str, model_path: str = 'baseline_lr_tuned.pkl'):
    texts, labels, _ = load_dataset(data_dir)
    emb = generate_embeddings(texts)
    le  = LabelEncoder()
    y   = le.fit_transform(labels)

    X_tr, X_val, y_tr, y_val = train_test_split(
        emb, y, test_size=0.2, random_state=42, stratify=y
    )
    grid = GridSearchCV(
        LogisticRegression(solver='lbfgs'),
        {'C': [0.01, 0.1, 1, 10], 'max_iter': [500, 1000]},
        cv=5, scoring='f1_micro'
    )
    grid.fit(X_tr, y_tr)

    best = grid.best_estimator_
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best, 'encoder': le}, f)

    preds = best.predict(X_val)

    # dict for dataframe
    report_dict = classification_report(
        y_val, preds,
        target_names=le.classes_,
        zero_division=0,
        output_dict=True
    )
    # raw text
    report_str = classification_report(
        y_val, preds,
        target_names=le.classes_,
        zero_division=0
    )
    acc = accuracy_score(y_val, preds)

    return report_dict, report_str, acc



def train_transformer(data_dir: str, model_dir: str = 'best_transformer', epochs: int = 3):
    texts, labels, _ = load_dataset(data_dir)
    le   = LabelEncoder(); y = le.fit_transform(labels)
    t_texts, v_texts, t_y, v_y = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    tok = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_ds = ResumeDataset(t_texts, t_y, tok)
    val_ds   = ResumeDataset(v_texts, v_y, tok)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))
    args  = TrainingArguments(output_dir=model_dir, learning_rate=5e-5, per_device_train_batch_size=8,
                                per_device_eval_batch_size=8, num_train_epochs=epochs, weight_decay=0.01)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    with open(f"{model_dir}/label_encoder.pkl", 'wb') as f: pickle.dump(le, f)
    return metrics


def match_resume_to_jobs(pdf_path: str, jd_embs: dict, top_k: int = 5):
    raw  = extract_text_from_pdf(pdf_path)
    proc = preprocess_text(raw)
    emb  = generate_embeddings([proc])[0].reshape(1, -1)
    sims = {cat: cosine_similarity(emb, jd.reshape(1,-1))[0][0] for cat, jd in jd_embs.items()}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]


def predict_and_match(pdf_path: str, model_path: str = 'baseline_lr_tuned.pkl', jd_csv: str = 'job_descriptions.csv'):
    base_data = pickle.load(open(model_path,'rb'))
    base_model, base_enc = base_data['model'], base_data['encoder']
    tok, tr_model, tr_enc = load_transformer_resources()
    jd_df = load_job_descriptions(jd_csv)

    raw  = extract_text_from_pdf(pdf_path)
    proc = preprocess_text(raw)

    emb     = generate_embeddings([proc])
    bl_cat  = base_enc.inverse_transform(base_model.predict(emb))[0]
    enc     = tok([raw], truncation=True, padding=True, return_tensors='pt')
    tr_cat  = tr_enc.inverse_transform(torch.argmax(tr_model(**enc).logits, axis=1).numpy())[0]
    skills  = extract_skills(raw)
    jd_embs = dict(zip(jd_df['category'], generate_embeddings(jd_df['description'].fillna('').apply(preprocess_text))))
    top5    = sorted({c: cosine_similarity(emb, e.reshape(1,-1))[0][0] for c,e in jd_embs.items()}.items(), key=lambda x: x[1], reverse=True)[:5]
    return bl_cat, tr_cat, skills, top5

# Streamlit UI
st.title('Smart Hiring Resume Screening')

mode = st.sidebar.selectbox('Mode', ['Train & Evaluate', 'Predict'])

if mode == 'Train & Evaluate':
    st.header('📊 Training & Evaluation')
    if st.button('🚀 Run Training'):
        with st.spinner('Training models...'):
            # SBERT+LR baseline
            report_dict, report_str, acc = train_classifier('data/resumes')

            # render as a DataFrame
            report_df = pd.DataFrame(report_dict).transpose()
            report_df.index.name = 'Category'
            report_df = report_df.rename(columns={
                'precision': 'Precision',
                'recall': 'Recall',
                'f1-score': 'F1 Score',
                'support': 'Support'
            })
            report_df = report_df[['Precision', 'Recall', 'F1 Score', 'Support']].round(3)
        st.success('✅ Baseline SBERT+LR complete!')
        # Show baseline metrics
        base_col1, base_col2 = st.columns(2)
        base_col1.metric('Baseline Accuracy', f'{acc:.3f}')
        # Transformer training
        with st.spinner('Training DistilBERT...'):
            tr_metrics = train_transformer('data/resumes')
        st.success('✅ DistilBERT training complete!')
        # Display transformer metrics in four columns
        eval_acc = tr_metrics.get('eval_accuracy', 0.0)
        eval_prec = tr_metrics.get('eval_precision', 0.0)
        eval_rec = tr_metrics.get('eval_recall', 0.0)
        eval_f1 = tr_metrics.get('eval_f1', 0.0)

        tr_cols = st.columns(4)
        # tr_cols[0].metric('Accuracy',    f"{tr_metrics['accuracy']:.3f}")
        # tr_cols[1].metric('Precision',   f"{tr_metrics['precision']:.3f}")
        # tr_cols[2].metric('Recall',      f"{tr_metrics['recall']:.3f}")
        # tr_cols[3].metric('F1 Score',    f"{tr_metrics['f1']:.3f}")
        tr_cols[0].metric('Accuracy', f"{eval_acc:.3f}")
        tr_cols[1].metric('Precision', f"{eval_prec:.3f}")
        tr_cols[2].metric('Recall', f"{eval_rec:.3f}")
        tr_cols[3].metric('F1 Score', f"{eval_f1:.3f}")
        st.markdown('---')
        st.subheader('📝 Baseline Classification Report')
        st.dataframe(report_df, height=300)

elif mode == 'Predict':
    st.header('🔍 Resume Prediction')
    uploaded = st.file_uploader('Upload Resume (PDF)', type='pdf')
    if uploaded:
        with st.spinner('Analyzing resume...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpf:
                tmpf.write(uploaded.read())
                pdf_path = tmpf.name
            bl_cat, tr_cat, skills, top5 = predict_and_match(pdf_path)
            os.remove(pdf_path)
        st.success('✅ Analysis complete!')
        st.subheader('🏷️ Predicted Categories')
        cat_col1, cat_col2 = st.columns(2)
        cat_col1.markdown(f"<p style='font-size:14px; margin:0'>SBERT+LR: <strong>{bl_cat}</strong></p>",unsafe_allow_html=True)
        cat_col2.markdown(f"<p style='font-size:14px; margin:0'>DistilBERT: <strong>{tr_cat}</strong></p>",unsafe_allow_html=True)
        st.markdown('---')
        st.subheader('💡 Extracted Skills')
        if skills:
            skill_cols = st.columns(3)
            for idx, skill in enumerate(skills):
                skill_cols[idx % 3].write(f'- {skill}')
        else:
            st.write('None detected')
        st.markdown('---')
        st.subheader('💼 Top 5 Job Matches')
        df_matches = pd.DataFrame(top5, columns=['Category','Score'])
        st.dataframe(df_matches.style.format({'Score': '{:.3f}'}), height=210)
