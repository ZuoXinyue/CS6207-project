import torch
import logging


from datasets import load_dataset
from utils import load_args, load_model, load_from_split_database, index_database, dataset_2_dataloader, clean, cluster_embeddings_with_faiss, cluster_embeddings_with_dbscan,cluster_embeddings_with_hierarchical, save_model
from trainer import train_RAG, test_RAG, val_RAG
from transformers import AdamW
from autoencoder import Autoencoder
import os
import tqdm
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

            

def main():
    args = load_args()
    logger.info(f"args: {args}")

    # load model
    tokenizer, model = load_model(args)
    model.config.question_encoder.max_position_embeddings = args.max_input_length
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # load database
    database = load_from_split_database(args.vec_database_path, args.init_database_name)
    # indexing the database
    corpus = database["text"]
    # dataset_emb = np.array()
    # print("database['embeddings']",database['embeddings'][0])
    
    # embeddings = torch.tensor(database['embeddings'])
    # faissIndex = index_database(embeddings)
    
    embeddings = torch.tensor(database['embeddings']).numpy()
    # Clustering embeddings
    
    if args.cluster == 'DBSCAN':
        cluster_centers, indexes,cluster_global_indices = cluster_embeddings_with_dbscan(embeddings, args.n_clusters)
    elif args.cluster == 'hierarchical':
        cluster_centers, indexes,cluster_global_indices = cluster_embeddings_with_hierarchical(embeddings, args.n_clusters)
    else:
        cluster_centers, indexes,cluster_global_indices = cluster_embeddings_with_faiss(embeddings, args.n_clusters)
        

    
    if args.debug_mode:
        dataset_train = clean(load_dataset(args.dataset_name, split='train[:50]' if args.debug_model else 'train'))
        dataset_val = clean(load_dataset(args.dataset_name, split='validation[:50]' if args.debug_model else 'validation'))
    else:
        # load dataset
        dataset_train = clean(load_dataset(args.dataset_name, split='train' if args.debug_model else 'train'))
        dataset_val = clean(load_dataset(args.dataset_name, split='validation' if args.debug_model else 'validation'))
    # dataset_test = load_dataset(args.dataset_name, split='test[:500]' if args.debug_model else 'test')
    logger.info(f"# Training samples: {len(dataset_train)}")
    logger.info(f"# Validation samples: {len(dataset_val)}")
    # logger.info(f"# Testing samples: {len(dataset_test)}")
    
    dataloader_train = dataset_2_dataloader(dataset_train, tokenizer, True, args)
    dataloader_val = dataset_2_dataloader(dataset_val, tokenizer, False, args)
    
    # load autoencoder
    autoencoder = Autoencoder(args.input_dim, args.latent_dim, args.device)
    for epoch in range(args.epoch_num):
        # 1. Train a RAG langauge model
        train_RAG(dataloader_train, model, tokenizer, optimizer, epoch, corpus, cluster_centers, indexes, cluster_global_indices, args)
        val_RAG(dataloader_val, model, tokenizer, epoch, corpus, cluster_centers, indexes,cluster_global_indices,args)
        
        # 2. Train an auto-encoder
        autoencoder.train_model(dataloader_train, dataloader_val, model, epoch, args)
        
        # 3. Update database
        save_model(epoch, model, autoencoder, args)


def test(model_path):
    args = load_args()
    logger.info(f"args: {args}")

    # load model
    tokenizer, model = load_model(args)
    # model.load_state_dict(torch.load(model_path))
    model.config.question_encoder.max_position_embeddings = args.max_input_length
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # load database
    database = load_from_split_database(args.vec_database_path, args.init_database_name)
    # indexing the database
    corpus = database["text"]
    
    embeddings = torch.tensor(database['embeddings']).numpy()
    # Clustering embeddings
    
    if args.cluster == 'DBSCAN':
        cluster_centers, indexes,cluster_global_indices = cluster_embeddings_with_dbscan(embeddings, args.n_clusters)
    elif args.cluster == 'hierarchical':
        cluster_centers, indexes,cluster_global_indices = cluster_embeddings_with_hierarchical(embeddings, args.n_clusters)
    else:
        cluster_centers, indexes,cluster_global_indices = cluster_embeddings_with_faiss(embeddings, args.n_clusters)

    if args.debug_mode:
        dataset_train = clean(load_dataset(args.dataset_name, split='train[:10]' if args.debug_mode else 'train'))
        dataset_val = clean(load_dataset(args.dataset_name, split='validation[:10]' if args.debug_mode else 'validation'))
    else:
        # load dataset
        dataset_train = clean(load_dataset(args.dataset_name, split='train' if args.debug_mode else 'train'))
        dataset_val = clean(load_dataset(args.dataset_name, split='validation' if args.debug_mode else 'validation'))
        
    logger.info(f"# Validation samples: {len(dataset_train)}")
    dataloader_val = dataset_2_dataloader(dataset_train, tokenizer, False, args)
    
    predictions = test_RAG(dataloader_val, model, tokenizer, 1, corpus, cluster_centers, indexes,cluster_global_indices,args)
    print("predictions",predictions)
    
    # EM between prediction & ground trut       
    golds = [ex['answers'] for ex in dataset_val]
    em_count = 0
    for pred, gold in zip(predictions, golds):
        if pred.strip() in gold:
            em_count += 1
    print(f"Exact Match: {em_count}/{len(predictions)}")
    
    # bleu score between prediction & ground truth
    

    
if __name__ == "__main__":
    # main()
    test('/home/yifan/projects/CS6207/CS6207-project/code/results/rag_model_DBSCAN_epoch_2.bin')