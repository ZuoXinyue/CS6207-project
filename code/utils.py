import os
import torch
import faiss
import warnings
from tqdm import tqdm
from typing import Union
from datasets import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from argparse import ArgumentParser
from enum import Enum, EnumMeta
from faiss import Kmeans, IndexFlatL2
import numpy as np
import logging
from sklearn.cluster import DBSCAN, AgglomerativeClustering


warnings.filterwarnings('ignore')

def load_args():
    parser = ArgumentParser()
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="The device to run the model")
    parser.add_argument("--model_name", type=str, default="facebook/rag-token-nq", help="The name of the model")
    parser.add_argument('--dataset_name', type=str, default="rajpurkar/squad_v2", help="The name of dataset from huggingface")
    parser.add_argument('--vec_database_path', type=str, default="../database_embed", help="The path of the vectorized database")
    parser.add_argument("--init_database_name", type=str, default="initial_retrieve_database", help="The name of the initial database")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size for training")
    parser.add_argument("--n_docs", type=int, default=5, help="The number of documents to retrieve")
    parser.add_argument("--max_input_length", type=int, default=256, help="The maximum length of input")
    parser.add_argument("--max_output_length", type=int, default=64, help="The maximum length of input")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="The learning rate for training")
    parser.add_argument("--epoch_num", type=int, default=1, help="The number of epochs for training")
    parser.add_argument("--debug_model", type=bool, default=True, help="Whether to use a debug model")
    parser.add_argument("--input_dim", type=int, default=768, help="The input dimension of the autoencoder")
    parser.add_argument("--latent_dim", type=int, default=128, help="The latent dimension of the autoencoder")
    parser.add_argument("--num_relevant_clusters", type=int, default=1, help="Number of relevant clusters to retrieve")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of Cluster")
    parser.add_argument("--cluster", type=str, default='kmeans', choices=["kmeans",'DBSCAN',"hierarchical"],help="Cluster Strategy")
    parser.add_argument('--debug_mode', action='store_true',
                        help="debug mode")
    
    
    
    args = parser.parse_args()
    return args

def clean(dataset):
    new_dataset = []
    for i in range(len(dataset)):
        try:
            if type(dataset[i]["answers"]["text"][0]) is str:
                new_dataset.append(dataset[i])
        except:
            continue
            
    return new_dataset

def dataset_2_dataloader(dataset, tokenizer, shuffle: bool, args: ArgumentParser) -> DataLoader:
    tensor_dataset_input = tokenizer([sample["question"] for sample in dataset], padding='max_length', truncation=True, max_length=args.max_input_length, return_tensors='pt')
    
    tensor_dataset_output = tokenizer.generator([sample["answers"]["text"][0] for sample in dataset], 
                                      padding='max_length', 
                                      truncation=True, 
                                    #   add_special_tokens=True,
                                      max_length=args.max_output_length, 
                                      return_tensors='pt')
    
    dataset = []
    for i in range(len(tensor_dataset_input["input_ids"])):
        dataset.append([
            tensor_dataset_input["input_ids"][i],
            tensor_dataset_input["attention_mask"][i],
            tensor_dataset_output["input_ids"][i]
        ])
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return dataloader

def index_database(embeddings: torch.tensor): 
    _, dim = embeddings.shape
    faissIndex = faiss.IndexFlatL2(dim)  # L2 distance calcutate similarity 
    faissIndex.add(embeddings.cpu().numpy()) 
    return faissIndex

def load_model(args: ArgumentParser) -> tuple[RagTokenizer, RagTokenForGeneration]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RagTokenForGeneration.from_pretrained(args.model_name)
    tokenizer = RagTokenizer.from_pretrained(args.model_name)
    
    model.to(device)

    return tokenizer, model
    
def get_embedding(input_text: Union[str, list[str]], tokenizer: RagTokenizer, model: RagTokenForGeneration) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dict = tokenizer.question_encoder(input_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    dataset = [[id, mask] for id, mask in zip(input_dict["input_ids"], input_dict["attention_mask"])]
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    all_embeddings = []
    for batch in dataloader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        batch_embeddings = model.question_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        all_embeddings.append(batch_embeddings.detach().cpu())
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings


def split_text_into_hunks(text: str, tokenizer: RagTokenizer) -> list[str]:
    tokens = tokenizer.question_encoder(text, add_special_tokens=False).input_ids
    hunks = []
    for i in range(0, len(tokens), 510):
        hunk = tokenizer.question_encoder.convert_ids_to_tokens(tokens[i:i+510])
        hunks.append(" ".join(hunk))
    return hunks
        
def database_embed(database_path: str, tokenizer: RagTokenizer, model: RagTokenForGeneration) -> Dataset:
    # get all file names in database_path
    file_names = [f for f in os.listdir(database_path) if os.path.isfile(os.path.join(database_path, f))]
    
    dataset = []
    for file_name in tqdm(file_names):
        with open(os.path.join(database_path, file_name), 'r') as f:
            text = f.read()
        
        # split text into hunks (each of 512 tokens)
        hunks = split_text_into_hunks(text, tokenizer)
        embeddings = get_embedding(hunks, tokenizer, model)
        for hunk, embedding in zip(hunks, embeddings):
            dataset.append({'title': file_name, 'text': hunk, 'embeddings': embedding})
    
    return Dataset.from_list(dataset)

def split_database(database: Dataset, bin_num: int) -> list[Dataset]:
    num_samples = len(database)
    bin_size = num_samples // bin_num
    datasets = []
    for i in range(bin_num):
        start = i * bin_size
        end = (i+1) * bin_size
        if i == bin_num - 1:
            end = num_samples
        small_dataset = [database[idx] for idx in range(start, end)]
        datasets.append(Dataset.from_list(small_dataset))
    return datasets

def load_from_split_database(split_database_path: str, database_name: str) -> Dataset:
    # find all files start with database_name in split_database_path
    file_names = [f for f in os.listdir(split_database_path) if f.startswith(database_name)]
    file_names = sorted(file_names)
    datasets = []
    for file_name in file_names:
        dataset = torch.load(os.path.join(split_database_path, file_name))
        datasets.extend([dataset[idx] for idx in range(len(dataset))])
    dataset = Dataset.from_list(datasets)
    return dataset
    
def make_initial_database():
    tokenizer, model = load_model(without_retriever=True)
    initial_dataset = database_embed("../database_text", tokenizer, model)
    # split dataset into 3 parts
    split_databases = split_database(initial_dataset, 3)
    for i, small_database in enumerate(split_databases):
        torch.save(small_database, f"../database_embed/initial_retrieve_database_{i}.pt")

def fill_from_dict(defaults, a_dict):
    for arg, val in a_dict.items():
        d = defaults.__dict__[arg]
        if type(d) is tuple:
            d = d[0]
        if isinstance(d, Enum):
            defaults.__dict__[arg] = type(d)[val]
        elif isinstance(d, EnumMeta):
            defaults.__dict__[arg] = d[val]
        else:
            defaults.__dict__[arg] = val

def fill_from_args(defaults):
    """
    Builds an argument parser, parses the arguments, updates and returns the object 'defaults'
    :param defaults: an object with fields to be filled from command line arguments
    :return:
    """
    parser = ArgumentParser()
    # if defaults has a __required_args__ we set those to be required on the command line
    required_args = []
    if hasattr(defaults, '__required_args__'):
        required_args = defaults.__required_args__
        for reqarg in required_args:
            if reqarg not in defaults.__dict__:
                raise ValueError(f'argument "{reqarg}" is required, but not present in __init__')
            if reqarg.startswith('_'):
                raise ValueError(f'arguments should not start with an underscore ({reqarg})')
    for attr, value in defaults.__dict__.items():
        # ignore members that start with '_'
        if attr.startswith('_'):
            continue

        # if it is a tuple, we assume the second is the help string
        help_str = None
        if type(value) is tuple and len(value) == 2 and type(value[1]) is str:
            help_str = value[1]
            value = value[0]

        # check if it is a type we can take on the command line
        if type(value) not in [str, int, float, bool] and not isinstance(value, Enum) and not isinstance(value, type):
            raise ValueError(f'Error on {attr}: cannot have {type(value)} as argument')
        if type(value) is bool and value:
            raise ValueError(f'Error on {attr}: boolean arguments (flags) must be false by default')

        # also handle str to enum conversion
        t = type(value)
        if isinstance(value, Enum):
            t = str
            value = value.name
        elif isinstance(value, EnumMeta):
            t = type
            value = str

        if t is type:
            # indicate a required arg by specifying a type rather than value
            parser.add_argument('--'+attr, type=value, required=True, help=help_str)
        elif t is bool:
            # support bool with store_true (required false by default)
            parser.add_argument('--'+attr, default=False, action='store_true', help=help_str)
        else:
            parser.add_argument('--'+attr, type=t, default=value, help=help_str, required=(attr in required_args))
    args = parser.parse_args()
    # now update the passed object with the arguments
    fill_from_dict(defaults, args.__dict__)
    # call _post_argparse() if the method is defined
    try:
        defaults._post_argparse()
    except AttributeError:
        pass
    return defaults




def cluster_embeddings_with_faiss(embeddings, n_clusters=100):
    """
    Cluster embeddings using FAISS KMeans.
    
    Args:
    - embeddings: 2D numpy array of shape (num_documents, embedding_size).
    - n_clusters: The number of clusters to form.
    
    Returns:
    - kmeans: The FAISS KMeans object after fitting.
    - cluster_centers: The centroids of the clusters, numpy array of shape (n_clusters, embedding_size).
    - cluster_assignments: Index of the cluster each sample belongs to, numpy array of shape (num_documents,).
    """
    d = embeddings.shape[1]  # Dimension of each vector
    kmeans = faiss.Kmeans(d, n_clusters, nredo=10)
    kmeans.train(embeddings.astype(np.float32))
    
    # The cluster centers (centroids)
    cluster_centers = kmeans.centroids
    # To get the cluster assignment for each document
    _, cluster_assignments = kmeans.index.search(embeddings.astype(np.float32), 1)
    

    
    # Create a FAISS index for each cluster
    indexes = {}    
    cluster_global_indices = {}
    
    for cluster_id in range(n_clusters):
        in_cluster = np.where(cluster_assignments == cluster_id)[0]
        
        # FAISS
        cluster_embeddings = embeddings[in_cluster]
        index = faiss.IndexFlatL2(d)
        index.add(cluster_embeddings)
        
        # local index and global index
        indexes[cluster_id] = index
        cluster_global_indices[cluster_id] = in_cluster
    
    return cluster_centers, indexes, cluster_global_indices


def cluster_embeddings_with_dbscan(embeddings, eps=0.5, min_samples=5):
    """
    Cluster embeddings using DBSCAN.
    
    Args:
    - embeddings: 2D numpy array of shape (num_documents, embedding_size).
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - cluster_centers: The centroids of the clusters, numpy array.
    - indexes: A dictionary of FAISS indexes for each cluster.
    - cluster_global_indices: A dictionary mapping cluster IDs to document indices in the original dataset.
    """
    print("Using DBSCAN...")
    # DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = clustering.labels_
    
    # cluster center
    unique_labels = set(labels)
    cluster_centers = []
    for label in unique_labels:
        if label == -1:
            continue  # ignore noise
        cluster_mask = (labels == label)
        cluster_embeddings = embeddings[cluster_mask]
        cluster_center = cluster_embeddings.mean(axis=0)
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    
    d = embeddings.shape[1]  # Dimension of each vector
    indexes = {}
    cluster_global_indices = {}
    
    for label in unique_labels:
        if label == -1:
            continue  # ignore noise
        in_cluster = np.where(labels == label)[0]
        
        cluster_embeddings = embeddings[in_cluster]
        index = faiss.IndexFlatL2(d)
        index.add(cluster_embeddings.astype(np.float32))
        
        indexes[label] = index
        cluster_global_indices[label] = in_cluster
    
    return cluster_centers, indexes, cluster_global_indices


def cluster_embeddings_with_hierarchical(embeddings, n_clusters=100):
    """
    Cluster embeddings using Hierarchical Clustering.
    
    Args:
    - embeddings: 2D numpy array of shape (num_documents, embedding_size).
    - n_clusters: The number of clusters to form.
    
    Returns:
    - cluster_centers: The centroids of the clusters, numpy array.
    - indexes: A dictionary of FAISS indexes for each cluster.
    - cluster_global_indices: A dictionary mapping cluster IDs to document indices in the original dataset.
    """
    print("Using hierarchical...")
    # hierarchical
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    labels = clustering.labels_
    
    # center
    unique_labels = set(labels)
    cluster_centers = np.array([embeddings[labels == label].mean(axis=0) for label in unique_labels])
    
    # FAISS
    d = embeddings.shape[1]  # Dimension of each vector
    indexes = {}
    cluster_global_indices = {}
    
    for label in unique_labels:
        in_cluster = np.where(labels == label)[0]
        
        cluster_embeddings = embeddings[in_cluster]
        index = faiss.IndexFlatL2(d)
        index.add(cluster_embeddings.astype(np.float32))
        
        indexes[label] = index
        cluster_global_indices[label] = in_cluster
    
    return cluster_centers, indexes, cluster_global_indices



def get_relevant_clusters(query_embeddings, cluster_centers, num_clusters=1):
    """
    Identify the most relevant cluster(s) for a query embedding.
    
    Args:
    - query_embeddings: The embedding of the query, shape (batch_size, embedding_dim).
    - cluster_centers: The centroids of the clusters, shape (num_clusters, embedding_dim).
    - num_clusters: Number of relevant clusters to retrieve.
    
    Returns:
    - Indices of the top `num_clusters` closest clusters to the query embedding.
    """
    # # Calculate distances from the query to each cluster center
    # distances = np.linalg.norm(cluster_centers - query_embedding, axis=1)
    # # Get the indices of the clusters with the smallest distances to the query
    # closest_clusters = np.argsort(distances)[:num_clusters]
    all_closest_clusters = []
    for query_embedding in query_embeddings:
        # Ensure query_embedding is 2D for consistent broadcasting
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate distances from the query to each cluster center
        distances = np.linalg.norm(cluster_centers - query_embedding, axis=1)
        
        # Get the indices of the clusters with the smallest distances to the query
        closest_clusters = np.argsort(distances)[:num_clusters]
        all_closest_clusters.append(closest_clusters)
    return all_closest_clusters


def save_model(epoch, model, autoencoder, args, save_dir="./results"):
    # dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save RAG
    model_save_path = os.path.join(save_dir, f"rag_model_{args.cluster}_epoch_{epoch}.bin")
    torch.save(model.state_dict(), model_save_path)
    # logger.info(f"RAG model saved to {model_save_path}")

    # save autoencoder
    autoencoder_save_path = os.path.join(save_dir, f"autoencoder_{args.cluster}_epoch_{epoch}.bin")
    torch.save(autoencoder.state_dict(), autoencoder_save_path)
    # logger.info(f"Autoencoder model saved to {autoencoder_save_path}")  