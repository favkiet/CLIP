from main import build_loaders, make_train_valid_dfs
from dataset import CLIPDataset, get_transforms
import pandas as pd
import torch
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
from CLIP import CLIPModel
from tqdm import tqdm
import torchvision
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import pickle
import streamlit as st


st.header("Image Search by Text",  divider='rainbow')
query = st.text_input("Input your query: ", placeholder=None)
n = st.number_input("Insert a number", value=None, placeholder="Type a number...", step=1)
button = st.button("Search")

model_path = "/Users/huynhanhkiet/Desktop/Computer_Vision/Image-Search/CLIP/model/best_CLIP.pt"
model = CLIPModel().to(torch.device("mps"))
model.load_state_dict(torch.load(model_path, map_location=torch.device("mps")))
model.eval()

df = pd.read_csv('/Users/huynhanhkiet/Desktop/Computer_Vision/data/captions.csv')
image_filenames = df['image'].values
embedding_path = '/Users/huynhanhkiet/Desktop/Computer_Vision/data/full_image_embeddings.pkl'
with open(embedding_path, 'rb') as f:
    image_embeddings = pickle.load(f)

if button:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(torch.device("mps"))
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    st.title(f"Top {n} Matches:")
    groups = []
    for i in range(0, len(matches), 3):
        groups.append(matches[i:i+3])
    cols = st.columns(3)
    for group in groups:
        for i, match in enumerate(group):
            image_path = f"/Users/huynhanhkiet/Desktop/Computer_Vision/data/Images/{match}"
            # image = cv2.imread(image_path)
            # image = cv2.res
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(image_path)
            image = image.resize((600, 600))
            cols[i].image(image, caption=match, use_column_width=True)
