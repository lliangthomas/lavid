import spacy
import numpy as np
import torch
import json
from clip import clip

def get_spacy_embeddings(words, model_name="en_core_web_md", embedding_dim=300):
    nlp = spacy.load(model_name)
    embeddings = np.zeros((len(words), embedding_dim))
    
    for i, word in enumerate(words):
        doc = nlp(word)
        embeddings[i] = doc.vector
        
    return embeddings

def get_clip_embeddings(words, save_file, model_name="ViT-B/32"):
    device = "cuda"
    model, preprocess = clip.load(model_name, device=device)
    
    text_tokens = clip.tokenize(words).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    
    embeddings = text_features.cpu().numpy()
    
    output_data = {}
    for i, word in enumerate(words):
        output_data[word] = embeddings[i].tolist()

    with open(save_file, "w", encoding="utf-8") as outf:
        json.dump(output_data, outf, ensure_ascii=False, indent=4)
    
    return embeddings

def get_minilm_embeddings(words, save_file, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(words, convert_to_numpy=True)

    output_data = {}
    for i, word in enumerate(words):
        output_data[word] = embeddings[i].tolist()

    with open(save_file, "w", encoding="utf-8") as outf:
        json.dump(output_data, outf, ensure_ascii=False, indent=4)

    return embeddings

def get_bert_embeddings(words, save_file, model_name="bert-base-uncased"):
    from transformers import BertModel, BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    model.eval()
    
    embeddings = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        word_embedding = outputs.last_hidden_state[:, 0, :].numpy().squeeze()
        embeddings.append(word_embedding)
    
    embeddings = np.array(embeddings)
    
    output_data = {}
    for i, word in enumerate(words):
        output_data[word] = embeddings[i].tolist()
    
    with open(save_file, "w", encoding="utf-8") as outf:
        json.dump(output_data, outf, ensure_ascii=False, indent=4)
    
    return embeddings