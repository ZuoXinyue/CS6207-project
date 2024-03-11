import torch
from transformers import RagConfig,RagTokenizer, RagRetriever, RagTokenForGeneration, Trainer, TrainingArguments,DataCollatorForSeq2Seq
from datasets import load_dataset
import faiss
from transformer_optimize import TransformerOptimize
from tqdm.auto import tqdm
"""
Combine Retrieval + Generation -> improve the performance.

Specifically,  when user raise a questions, 
RAG first use the retriver query external
database (a subset of Wikipedia) to fine the related document or text.
And then the retrieved texts will be used as an cotext, help model generate more accuract answers.
"""

from rag_hypers import RagHypers
hypers = RagHypers().fill_from_args()

tokenizer, model = hypers.get_tokenizer_and_model()
print("loaded")

model = model.to(hypers.device)
optimizer = TransformerOptimize(hypers, hypers.num_train_epochs * hypers.num_instances, model)
model.train()

##### load external database and FAISS index
initial_dataset_path = "../dataset/initial_retrieve_database.pt"
init_dataset = torch.load(initial_dataset_path)

context_embeddings = init_dataset['embeddings']
context_embeddings = torch.Tensor(context_embeddings)
num_texts, dim = context_embeddings.shape

faissIndex = faiss.IndexFlatL2(dim)  # L2距离用于向量相似性
faissIndex.add(context_embeddings.cpu().numpy())  # 向索引中添加向量
#文本数据
texts = init_dataset['text']

# 加载SQuAD数据集
squad_dataset = load_dataset("squad", split='train')

# 预处理SQuAD数据集
def preprocess_data(examples):
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    answers = [ans['text'][0] for ans in examples['answers']]  # 假设我们只关心第一个答案
    labels = tokenizer(answers, padding='max_length', truncation=True, max_length=512, return_tensors='pt')['input_ids']
    return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': labels}

squad_processed = squad_dataset.map(preprocess_data, batched=True,batch_size=200)



def prepare_Questions(batch):
    # 准备数据，对问题进行编码等
    inputs = tokenizer(batch['question'], padding=True, truncation=True, return_tensors="pt")
    return inputs


def retrieve(encoded_queries, answers):
    labels = tokenizer(answers, padding=True, truncation=True, return_tensors="pt", max_length=512)["input_ids"].to(model.device)
    input_dict = tokenizer.prepare_seq2seq_batch(encoded_queries, return_tensors="pt") 

    question_hidden_states = model.question_encoder(input_ids=input_dict["input_ids"])[0]
    question_hidden_states_np = question_hidden_states.cpu().detach().numpy()

    D, I = faissIndex.search(question_hidden_states_np,5)  # n_docs the document number
    print('D, I',D.shape, I.shape)
    passages = [init_dataset['text'][i] for i in I.flatten()]
    context_input_ids = []
    context_attention_mask = []
    for passage in passages:
        encoded_passage = tokenizer(passage, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        context_input_ids.append(encoded_passage['input_ids'].squeeze(0))  # 去掉批次维度
        context_attention_mask.append(encoded_passage['attention_mask'].squeeze(0))

    context_input_ids = torch.stack(context_input_ids).to(model.device)
    context_attention_mask = torch.stack(context_attention_mask).to(model.device)
    doc_scores = torch.tensor(D).to(model.device)  # 将FAISS返回的距离分数转换为tensor
    input_ids = input_dict['input_ids'].to(model.device)
    attention_mask = input_dict['attention_mask'].to(model.device)

    return context_input_ids, context_attention_mask, doc_scores, input_ids, attention_mask, labels




# model_path = "facebook/rag-token-nq"
# initial_dataset_path = "../dataset/initial_retrieve_database.pt"
def train():
    progress_bar = tqdm(squad_processed, desc="Training")
    for batch in squad_processed:
      
        queries = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        # answers = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        answers = [answer[0] for answer in batch['answers']]  # 假设我们只关心第一个答案
        # 假设retrieve函数接受编码后的输入和标签，并返回所需的所有tensor
        context_input_ids, context_attention_mask, doc_scores, input_ids, attention_mask, labels = retrieve(queries, answers)
        print('out')
        model.train()
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        labels=labels, 
                        context_input_ids=context_input_ids, 
                        context_attention_mask=context_attention_mask,
                        doc_scores=doc_scores)
        print('out2')
        
        optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        progress_bar.set_postfix({'loss': outputs.loss.item()})

train()
