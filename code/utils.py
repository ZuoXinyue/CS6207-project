import os
import torch
import warnings
from tqdm import tqdm
from typing import Union
from datasets import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from argparse import ArgumentParser
from enum import Enum, EnumMeta
import logging


warnings.filterwarnings('ignore')
def load_model(local_model_path: Union[str, None] = None, without_retriever: bool = False) -> Union[tuple[RagTokenizer, RagRetriever, RagTokenForGeneration], tuple[RagTokenizer, RagTokenForGeneration]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    if without_retriever:
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    else:
        if local_model_path:
            tokenizer = RagTokenizer.from_pretrained(local_model_path)
            retriever = RagRetriever.from_pretrained(local_model_path, index_name="exact", use_dummy_dataset=True)  # 根据你的设置调整
            model = RagTokenForGeneration.from_pretrained(local_model_path, retriever=retriever)
            
        else:
            initial_dataset = torch.load("../dataset_embed/initial_dataset.pt")
            retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", indexed_dataset=initial_dataset)  # 根据你的设置调整
            model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    
    model.to(device)
    if without_retriever:
        return tokenizer, model
    else:
        return tokenizer, retriever, model
    
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
    
from argparse import ArgumentParser
from enum import Enum, EnumMeta
import logging

logger = logging.getLogger(__name__)

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