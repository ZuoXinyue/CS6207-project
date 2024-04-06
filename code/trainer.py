import torch
from datasets import load_dataset
import faiss
from transformer_optimize import TransformerOptimize
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler 
from utils import get_relevant_clusters
import numpy as np
scaler = GradScaler()
# from finetune_peft import get_peft_config, PEFTArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
#######

def retrieve(query_embedding, tokenizer, corpus, cluster_centers,indexes,cluster_global_indices,args):
    
    batch_size = args.batch_size
    question_hidden_states_np = query_embedding.cpu().detach().numpy()
    # Identify the most relevant cluster(s) for the query
    relevant_clusters = get_relevant_clusters(query_embedding.cpu().detach().numpy(), cluster_centers, num_clusters=args.num_relevant_clusters)
    
    
    # D, I = faissIndex.search(question_hidden_states_np,args.n_docs)  # n_docs the document number
    all_doc_ids = []
    all_doc_scores = []
    
    for cluster_id in relevant_clusters:
        if cluster_id in indexes:
            D, I = indexes[cluster_id].search(question_hidden_states_np, args.n_docs)
            scores = 1.0 / (1.0 + D[0])
            all_doc_scores.extend(scores.tolist())
            # Translate local cluster indices back to global corpus indices
            for idx in I[0]:
                global_idx = cluster_global_indices[cluster_id][idx]
                all_doc_ids.append(global_idx)


    context_input_ids = []
    context_attention_mask = []
   
    for idx in all_doc_ids:  # Directly iterate over all_doc_ids
        doc = corpus[idx]  # Access the document directly using the index
        encoded_doc = tokenizer(doc, max_length=args.max_input_length, padding='max_length', truncation=True, return_tensors="pt")
        context_input_ids.append(encoded_doc['input_ids'])
        context_attention_mask.append(encoded_doc['attention_mask'])

    context_input_ids = torch.cat(context_input_ids, dim=0)
    context_attention_mask = torch.cat(context_attention_mask, dim=0)
    
    # doc score just use the 1 / distance as the doc scores
    # doc_scores = torch.tensor(all_doc_scores)
    doc_scores = torch.tensor(all_doc_scores).to(args.device) 
    if len(doc_scores.size()) == 1:
        doc_scores = doc_scores.unsqueeze(0)  # 如果是一维，增加一个批次维度
    # Prepare the context input IDs and attention mask for the filtered results
    

    return context_input_ids.to(args.device), \
            context_attention_mask.to(args.device), \
            doc_scores \

def train_RAG(dataloader, model, tokenizer, optimizer, epoch, corpus, cluster_centers, indexes,cluster_global_indices,args):
    model.train()
    progress_bar = tqdm(dataloader, desc="Epoch {} Train RAG".format(epoch))
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(args.device) for b in batch]

        question_hidden_states = model.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
        context_input_ids, context_attention_mask, doc_scores = retrieve(question_hidden_states, tokenizer, corpus, cluster_centers,indexes, cluster_global_indices,args)
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        labels=labels, 
                        context_input_ids=context_input_ids, 
                        context_attention_mask=context_attention_mask,
                        doc_scores=doc_scores)
        
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)

def val_RAG(dataloader, model, tokenizer, epoch, corpus, cluster_centers, indexes,cluster_global_indices,args):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Epoch {} Val RAG".format(epoch)):
            input_ids, attention_mask, labels = [b.to(args.device) for b in batch]

            question_hidden_states = model.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
            context_input_ids, context_attention_mask, doc_scores = retrieve(question_hidden_states, tokenizer, corpus,cluster_centers, indexes,cluster_global_indices, args)
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            labels=labels, 
                            context_input_ids=context_input_ids, 
                            context_attention_mask=context_attention_mask,
                            doc_scores=doc_scores)
            
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}, RAG Validation Loss: {avg_loss}')
