from utils import load_model

tokenizer, retriever, model = load_model(local_model_path='/home/yifan/projects/CS6207/rag-token-nq')
print('successfully loaded the model')