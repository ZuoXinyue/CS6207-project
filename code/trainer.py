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
target_modules = [
    "question_encoder.question_encoder.bert_model.encoder.layer.0.attention.self.query",
    # "question_encoder.question_encoder.bert_model.encoder.layer.*.attention.self.key",
    # "question_encoder.question_encoder.bert_model.encoder.layer.*.attention.self.value",
    ]

peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False,
            r=4,
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=target_modules  # Specify target modules here
        )
from rag_hypers import RagHypers
from utils import load_from_split_database
hypers = RagHypers().fill_from_args()

tokenizer, model = hypers.get_tokenizer_and_model()
model = get_peft_model(model, peft_config)
model = model.to(hypers.device)


print(hypers.device)
optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.num_instances, model)

model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size = model_parameters * 4  
model_size_MB = model_size / (1024 * 1024) 
print(f"model size: {model_size_MB:.2f} MB, Model parameters count: {model_parameters}")


##### load external database and FAISS index
initial_dataset_path = "../database_embed"
init_dataset = load_from_split_database(initial_dataset_path, "initial_retrieve_database")

context_embeddings = init_dataset['embeddings']
context_embeddings = torch.Tensor(context_embeddings)
num_texts, dim = context_embeddings.shape
faissIndex = faiss.IndexFlatL2(dim)  # L2 distance calcutate similarity 
faissIndex.add(context_embeddings.cpu().numpy())  


# load SQuAD dataset
squad_dataset = load_dataset("squad", split='train')

squad_dataset = squad_dataset.select(range(1))

print('train: ', len(squad_dataset))
squad_dataset_val = load_dataset("squad", split='validation')
# preprocess SQuAD dataset
MAX_LENGTH = 512
print("MAX_LENGTH",MAX_LENGTH)
def preprocess_data(examples,max_length=MAX_LENGTH):
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    answers = [ans['text'][0] for ans in examples['answers']]  # first answer
    labels = tokenizer(answers, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')['input_ids']
    return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': labels}
squad_processed = squad_dataset.map(preprocess_data, batched=True, batch_size=1)


def retrieve(encoded_queries, answers):

    labels = tokenizer(answers, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LENGTH)["input_ids"].to(model.device)
    input_dict = tokenizer.prepare_seq2seq_batch(encoded_queries, return_tensors="pt") 
    input_dict["input_ids"] = input_dict["input_ids"].to(model.device)
    question_hidden_states = model.question_encoder(input_ids=input_dict["input_ids"])[0]
    question_hidden_states_np = question_hidden_states.cpu().detach().numpy()

    D, I = faissIndex.search(question_hidden_states_np,5)  # n_docs the document number
    print('D, I',D.shape, I.shape)
    passages = [init_dataset['text'][i] for i in I.flatten()]
    context_input_ids = []
    context_attention_mask = []
    for passage in passages:
        encoded_passage = tokenizer(passage, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        context_input_ids.append(encoded_passage['input_ids'].squeeze(0))  # 去掉批次维度
        context_attention_mask.append(encoded_passage['attention_mask'].squeeze(0))

    context_input_ids = torch.stack(context_input_ids).to(model.device)
    context_attention_mask = torch.stack(context_attention_mask).to(model.device)
    doc_scores = torch.tensor(D).to(model.device)  # 将FAISS返回的距离分数转换为tensor
    input_ids = input_dict['input_ids'].to(model.device)
    attention_mask = input_dict['attention_mask'].to(model.device)

    return context_input_ids, context_attention_mask, doc_scores, input_ids, attention_mask, labels



def train():
    progress_bar = tqdm(squad_processed, desc="Training")
    for batch in squad_processed:
        queries = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        answers = [answer[0] for answer in batch['answers']]  # only input the first answer
        # retrieve function 
        context_input_ids, context_attention_mask, doc_scores, input_ids, attention_mask, labels = retrieve(queries, answers)
        model.train()
        print(context_input_ids.shape, context_attention_mask.shape, doc_scores.shape, input_ids.shape, attention_mask.shape, labels.shape)
        # optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        labels=labels, 
                        context_input_ids=context_input_ids, 
                        context_attention_mask=context_attention_mask,
                        doc_scores=doc_scores)
            loss = outputs.loss
        print('out2')
        scaler.scale(loss).backward()

        # outputs.loss.backward()
        scaler.update()
        # optimizer.step()
        progress_bar.set_postfix({'loss': outputs.loss.item()})

train()
