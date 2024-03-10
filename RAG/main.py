from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)

# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

print('success')
# # 示例：生成对一个问题的答案
# question = "who holds the record in 100m freestyle"
# input_dict = tokenizer.prepare_seq2seq_batch(question, return_tensors="pt") 
# generated = model.generate(input_ids=input_dict["input_ids"]) 
# print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 
# # model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
