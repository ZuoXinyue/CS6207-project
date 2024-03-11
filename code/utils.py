import os
import torch
import warnings
from tqdm import tqdm
from typing import Union
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

warnings.filterwarnings('ignore')
def load_model(local_model_path: Union[str, None] = None, without_retriever: bool = False) -> Union[tuple[RagTokenizer, RagRetriever, RagTokenForGeneration], tuple[RagTokenizer, RagTokenForGeneration]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    if without_retriever:
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    else:
        initial_dataset = torch.load("dataset/initial_dataset.pt")
        retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", indexed_dataset=initial_dataset)  # 根据你的设置调整
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    if local_model_path:
        # load local parameters
        model.load_state_dict(torch.load(local_model_path))
    
    model.to(device)
    if without_retriever:
        return tokenizer, model
    else:
        return tokenizer, retriever, model
    
def get_embedding(input_text: Union[str, list[str]], tokenizer: RagTokenizer, model: RagTokenForGeneration) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dict = tokenizer.question_encoder(input_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    dataset = [[id, mask] for id, mask in zip(input_dict["input_ids"], input_dict["attention_mask"])]
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    all_embeddings = []
    for batch in dataloader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        batch_embeddings = model.question_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        all_embeddings.append(batch_embeddings.detach().cpu())
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings

def split_text_into_hunks(text: str, tokenizer: RagTokenizer) -> list[str]:
    tokens = tokenizer.question_encoder(text, add_special_tokens=False).input_ids
    hunks = []
    for i in range(0, len(tokens), 510):
        hunk = tokenizer.question_encoder.convert_ids_to_tokens(tokens[i:i+510])
        hunks.append(" ".join(hunk))
    return hunks
        
def database_embed(database_path: str, tokenizer: RagTokenizer, model: RagTokenForGeneration) -> Dataset:
    # get all file names in database_path
    file_names = [f for f in os.listdir(database_path) if os.path.isfile(os.path.join(database_path, f))]
    
    dataset = []
    for file_name in tqdm(file_names):
        with open(os.path.join(database_path, file_name), 'r') as f:
            text = f.read()
        
        # split text into hunks (each of 512 tokens)
        hunks = split_text_into_hunks(text, tokenizer)
        embeddings = get_embedding(hunks, tokenizer, model)
        for hunk, embedding in zip(hunks, embeddings):
            dataset.append({'title': file_name, 'text': hunk, 'embeddings': embedding})
    
    return Dataset.from_list(dataset)

def make_initial_dataset():
    tokenizer, model = load_model(without_retriever=True)
    initial_dataset = database_embed("../database", tokenizer, model)  
    torch.save(initial_dataset, "../dataset/initial_retrieve_database.pt")