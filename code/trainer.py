import torch
from datasets import load_dataset
import faiss
from transformer_optimize import TransformerOptimize
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler 
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

def retrieve(query_embedding, tokenizer, corpus, faissIndex, args):
    question_hidden_states_np = query_embedding.cpu().detach().numpy()
    
    D, I = faissIndex.search(question_hidden_states_np,args.n_docs)  # n_docs the document number

    context_input_ids = []
    context_attention_mask = []
    for i in range(args.batch_size):
        doc_ids = I[i]  # 当前查询检索到的文档索引
        docs = [corpus[idx] for idx in doc_ids]  # the retrived doc
        for doc in docs:
            encoded_doc = tokenizer(doc, max_length=args.max_input_length, padding='max_length', truncation=True, return_tensors="pt")
            context_input_ids.append(encoded_doc['input_ids'])
            context_attention_mask.append(encoded_doc['attention_mask'])

    context_input_ids = torch.cat(context_input_ids, dim=0)
    context_attention_mask = torch.cat(context_attention_mask, dim=0)
    
    # doc score just use the 1 / distance as the doc scores
    doc_scores = 1.0 / (1.0 + torch.tensor(D))

    return context_input_ids.to(args.device), \
            context_attention_mask.to(args.device), \
            doc_scores.to(args.device), \

def train_RAG(dataloader, model, tokenizer, optimizer, epoch, corpus, faissIndex, args):
    model.train()
    progress_bar = tqdm(dataloader, desc="Epoch {} Train RAG".format(epoch))
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(args.device) for b in batch]

        question_hidden_states = model.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
        context_input_ids, context_attention_mask, doc_scores = retrieve(question_hidden_states, tokenizer, corpus, faissIndex, args)
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

def val_RAG(dataloader, model, tokenizer, epoch, corpus, faissIndex, args):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Epoch {} Val RAG".format(epoch)):
            input_ids, attention_mask, labels = [b.to(args.device) for b in batch]

            question_hidden_states = model.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
            context_input_ids, context_attention_mask, doc_scores = retrieve(question_hidden_states, tokenizer, corpus, faissIndex, args)
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            labels=labels, 
                            context_input_ids=context_input_ids, 
                            context_attention_mask=context_attention_mask,
                            doc_scores=doc_scores)
            
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}, RAG Validation Loss: {avg_loss}')
