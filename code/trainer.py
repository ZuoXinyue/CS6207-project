import torch
from datasets import load_dataset
import faiss
from transformer_optimize import TransformerOptimize
from tqdm.auto import tqdm
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

BATCH_SIZE=1
N_DOCS = 5
MAX_LENGTH = 128
COMBINED_MAX = 300

####### lora Setting
target_modules = [
    "question_encoder.question_encoder.bert_model.encoder.layer.0.attention.self.query",
    # "question_encoder.question_encoder.bert_model.encoder.layer.*.attention.self.key",
    # "question_encoder.question_encoder.bert_model.encoder.layer.*.attention.self.value",
    ]
peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False,
            r=4, #### rank
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=target_modules  # Specify target modules here
        )
####### 

from rag_hypers import RagHypers
from utils import load_from_split_database
hypers = RagHypers().fill_from_args()
tokenizer, model = hypers.get_tokenizer_and_model()
# model = get_peft_model(model, peft_config)
model = model.to(hypers.device)
model.config.question_encoder.max_position_embeddings = MAX_LENGTH  # 限制生成回答的长度
optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.num_instances, model)

# print('device',hypers.device)
# model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# model_size = model_parameters * 4  
# model_size_MB = model_size / (1024 * 1024) 
# print(f"model size: {model_size_MB:.2f} MB, Model parameters count: {model_parameters}")


##### load external database and FAISS index
initial_dataset_path = "../database_embed"
init_dataset = load_from_split_database(initial_dataset_path, "initial_retrieve_database")

context_embeddings = init_dataset['embeddings']
context_texts = init_dataset['text']
context_embeddings = torch.Tensor(context_embeddings)
num_texts, dim = context_embeddings.shape
faissIndex = faiss.IndexFlatL2(dim)  # L2 distance calcutate similarity 
faissIndex.add(context_embeddings.cpu().numpy()) 
#########  



####### load SQuAD dataset
squad_dataset = load_dataset("squad", split='train')
squad_dataset = squad_dataset.select(range(100))
squad_dataset_val = load_dataset("squad", split='validation')
# preprocess SQuAD dataset
def preprocess_data(examples,max_length=MAX_LENGTH):
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    answers = [ans['text'][0] for ans in examples['answers']]  # first answer
    labels = tokenizer(answers, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')['input_ids']
    return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': labels}
squad_processed = squad_dataset.map(preprocess_data, batched=True, batch_size=BATCH_SIZE)
#######

def retrieve(input_ids, attention_mask):
    question_hidden_states = optimizer.model.module.question_encoder(input_ids=input_ids,attention_mask=attention_mask)[0]
    question_hidden_states_np = question_hidden_states.cpu().detach().numpy()
    
    D, I = faissIndex.search(question_hidden_states_np,N_DOCS)  # n_docs the document number

    context_input_ids = []
    context_attention_mask = []
    for i in range(BATCH_SIZE):
        doc_ids = I[i]  # 当前查询检索到的文档索引
        docs = [context_texts[idx] for idx in doc_ids]  # the retrived doc
        for doc in docs:
            encoded_doc = tokenizer(doc, max_length=COMBINED_MAX, padding='max_length', truncation=True, return_tensors="pt")
            context_input_ids.append(encoded_doc['input_ids'])
            context_attention_mask.append(encoded_doc['attention_mask'])

    context_input_ids = torch.cat(context_input_ids, dim=0)
    context_attention_mask = torch.cat(context_attention_mask, dim=0)
    
    # doc score just use the 1 / distance as the doc scores
    doc_scores = 1.0 / (1.0 + torch.tensor(D))


    return context_input_ids.to(model.device), \
            context_attention_mask.to(model.device), \
            doc_scores.to(model.device), \



def train():
    optimizer.model.train()
    for epoch in range(10):
        progress_bar = tqdm(squad_processed, desc="Epoch {} Training".format(epoch))
        for batch in squad_processed:
            #### load input_ids, attention_mask, labels from training data
            input_ids = torch.tensor(batch['input_ids'], dtype=torch.long).reshape(BATCH_SIZE, MAX_LENGTH).to(model.device)
            attention_mask = torch.Tensor(batch['attention_mask']).reshape(BATCH_SIZE, MAX_LENGTH).to(model.device)
            labels = torch.tensor(batch['labels'], dtype=torch.long).reshape(BATCH_SIZE, MAX_LENGTH).to(model.device)

            ##### start retrive ######
            context_input_ids, context_attention_mask, doc_scores = retrieve(input_ids, attention_mask)
            outputs = optimizer.model.module(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            labels=labels, 
                            context_input_ids=context_input_ids, 
                            context_attention_mask=context_attention_mask,
                            doc_scores=doc_scores)
            # outputs.loss.mean().backward()
            optimizer.step_loss(outputs.loss.mean())
            # loss = outputs.loss
            # scaler.scale(loss).backward()  # 使用 GradScaler 来缩放损失，然后进行反向传播
            # scaler.step(optimizer)  # 使用 GradScaler 来执行优化器的步骤
            # scaler.update()  # 更新 GradScaler
            # # optimizer.step()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': outputs.loss.item()})

train()
