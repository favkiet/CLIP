import config as CFG
import torch
from transformers import DistilBertTokenizer
from main import build_loaders, make_train_valid_dfs
from CLIP import CLIPModel
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

if __name__=='__main__':
    df = pd.read_csv('/Users/huynhanhkiet/Desktop/Computer_Vision/data/captions.csv')
    model, image_embeddings = get_image_embeddings(df[:10], "/Users/huynhanhkiet/Desktop/Computer_Vision/Image-Search/CLIP/model/best_CLIP.pt")
    df['embedding'] = None
    for idx, batch in enumerate(image_embeddings):
        df.at[idx, 'embedding'] = batch.cpu().detach().numpy()
    df.to_parquet('/Users/huynhanhkiet/Desktop/Computer_Vision/data_embedding/English/df_image_embedding_1.parquet.gzip', compression='gzip')
    df.to_pickle('/Users/huynhanhkiet/Desktop/Computer_Vision/data_embedding/English/df_image_embedding_1.pkl')