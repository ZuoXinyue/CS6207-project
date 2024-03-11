from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
# 请确保'./rag_model'是包含'pytorch_model.bin'和'config.json'的目录路径
model_path = "facebook/rag-token-nq"
tokenizer = RagTokenizer.from_pretrained(model_path)
# retriever = RagRetriever.from_pretrained(model_path, index_name="exact", use_dummy_dataset=True)  # 根据你的设置调整
model = RagTokenForGeneration.from_pretrained(model_path)
# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

print('success loaded th model')
# # 示例：生成对一个问题的答案
question = ["who holds the record in 100m freestyle", "who holds the record in 100m"]
input_dict = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt") 
print('input_dict: ',input_dict)
# generated = model.generate(input_ids=input_dict["input_ids"]) 
# print('tokenizer: ',tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 


outputs = model.question_encoder(input_ids=input_dict["input_ids"])[0]  # 获取问题编码器的输出
embeddings = outputs  # 这里的outputs就是输入文本的嵌入表示
print('embeddings: ',embeddings) 
print(f"embeddings shape: {embeddings.shape}")  # torch.Size([1, 768])







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