from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
# 请确保'./rag_model'是包含'pytorch_model.bin'和'config.json'的目录路径
model_path = "/home/yifan/projects/CS6207/rag-token-nq"
tokenizer = RagTokenizer.from_pretrained(model_path)
retriever = RagRetriever.from_pretrained(model_path, index_name="exact", use_dummy_dataset=True)  # 根据你的设置调整
model = RagTokenForGeneration.from_pretrained(model_path, retriever=retriever)
# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

print('success loaded th model')
# # 示例：生成对一个问题的答案
question = "who holds the record in 100m freestyle"
input_dict = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt") 
generated = model.generate(input_ids=input_dict["input_ids"]) 
print('embeddings: ',tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 







#### 
# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# # retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True, trust_remote_code=True)
# retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True, trust_remote_code=True)

# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# print('success')
# # # 示例：生成对一个问题的答案
# question = "who holds the record in 100m freestyle"
# print('success2222')
# input_dict = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt") 
# print('success233322')
# generated = model.generate(input_ids=input_dict["input_ids"]) 
# print(' ================== ')
# print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 
# # model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# print('success')