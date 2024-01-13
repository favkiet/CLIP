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
import pickle 


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
    
if __name__ == "__main__":
    # _, valid_df = make_train_valid_dfs()
    # model, image_embeddings = get_image_embeddings(valid_df, "/Users/huynhanhkiet/Desktop/Computer_Vision/Image-Search/CLIP/model/best_CLIP.pt")
    df = pd.read_csv('/Users/huynhanhkiet/Desktop/Computer_Vision/data/captions.csv')
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load("/Users/huynhanhkiet/Desktop/Computer_Vision/Image-Search/CLIP/model/best_CLIP.pt", map_location=CFG.device))
    model.eval()
    
    embedding_path = "/Users/huynhanhkiet/Desktop/Computer_Vision/data/full_image_embeddings.pkl"
    with open(embedding_path, 'rb') as f:
        image_embeddings = pickle.load(f)
    find_matches(model, image_embeddings, query="dogs on the grass", image_filenames=df['image'].values, n=9)
    

    