from main import build_loaders, make_train_valid_dfs
from dataset import CLIPDataset, get_transforms
import pandas as pd
import torch
from transformers import DistilBertTokenizer
import albumentations as A
import matplotlib.pyplot as plt
from CLIP import CLIPModel
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import torch.nn.functional as F
import pickle
import config as CFG

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
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
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()

if __name__=="__main__":
    df_embedding = pd.read_parquet('/Users/huynhanhkiet/Desktop/Computer_Vision/data_embedding/English/df_image_embedding.parquet.gzip')
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load("/Users/huynhanhkiet/Desktop/Computer_Vision/Image-Search/CLIP/model/best_CLIP.pt", map_location=CFG.device))
    model.eval()
    
    embedding = []
    for value in df_embedding['embedding'].values:
        embedding.append(torch.from_numpy(value).to(torch.device('mps')))
    image_embedding = torch.stack(embedding)
    find_matches(model, image_embedding, query="dogs on the grass", image_filenames=df_embedding['image'].values, n=9)
    
    