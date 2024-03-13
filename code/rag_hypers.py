import signal

import torch
import os
from transformers import RagTokenizer, RagSequenceForGeneration, RagTokenForGeneration
import logging
import torch
import os
import socket
from utils import fill_from_args, fill_from_dict
import ujson as json
import time
import random
import numpy as np


logger = logging.getLogger(__name__)


def dist_initialize():
    """
    initializes torch distributed
    :return: local_rank, global_rank, world_size
    """
    if "RANK" not in os.environ:
        local_rank = -1
        global_rank = 0
        world_size = 1
    else:
        if torch.cuda.device_count() == 0:
            err = f'No CUDA on {socket.gethostname()}'
            logger.error(err)
            raise ValueError(err)
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        env_master_addr = os.environ['MASTER_ADDR']
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if env_master_addr.startswith('file://'):
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method=env_master_addr,
                                                 world_size=world_size,
                                                 rank=global_rank)
            logger.info("init-method file: {}".format(env_master_addr))
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            torch.distributed.init_process_group(backend='nccl')
            logger.info("init-method master_addr: {} master_port: {}".format(
                env_master_addr, os.environ['MASTER_PORT']))
            local_rank = int(global_rank % torch.cuda.device_count())
    cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NOT SET'
    logger.info(f"world_rank {global_rank} cuda_is_available {torch.cuda.is_available()} "
                f"cuda_device_cnt {torch.cuda.device_count()} on {socket.gethostname()},"
                f" CUDA_VISIBLE_DEVICES = {cuda_devices}")
    return local_rank, global_rank, world_size


class HypersBase:
    """
    This should be the base hyperparameters class, others should extend this.
    """
    def __init__(self):
        self.local_rank, self.global_rank, self.world_size = dist_initialize()
        # required parameters initialized to the datatype
        self.model_type = ''
        self.model_name_or_path = ''
        self.resume_from = ''  # to resume training from a checkpoint
        self.config_name = ''
        self.tokenizer_name = ''
        self.cache_dir = ''
        self.do_lower_case = False
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0  # previous default was 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_instances = 0  # previous default was 0.1 of total
        self.num_train_epochs = 3
        self.no_cuda = False
        self.n_gpu = 1
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'  # previous default was O2
        self.full_train_batch_size = 1  # previous default was 32
        self.per_gpu_eval_batch_size = 1
        self.output_dir = ''  # where to save model
        self.save_total_limit = 1  # limit to number of checkpoints saved in the output dir
        self.save_steps = 0  # do we save checkpoints every N steps? (TODO: put in terms of hours instead)
        self.use_tensorboard = False
        self.log_on_all_nodes = False
        self.server_ip = ''
        self.server_port = ''
        # self.__required_args__ = ['model_type', 'model_name_or_path']

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def set_gradient_accumulation_steps(self):
        """
        when searching for full_train_batch_size in hyperparameter tuning we need to update
        the gradient accumulation steps to stay within GPU memory constraints
        :return:
        """
        if self.n_gpu * self.world_size * self.per_gpu_train_batch_size > self.full_train_batch_size:
            self.per_gpu_train_batch_size = self.full_train_batch_size // (self.n_gpu * self.world_size)
            self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = self.full_train_batch_size // \
                                               (self.n_gpu * self.world_size * self.per_gpu_train_batch_size)

    def _basic_post_init(self):
        # Setup CUDA, GPU
        if self.local_rank == -1 or self.no_cuda:
            # NOTE: changed "cuda" to "cuda:0"
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1

        if self.n_gpu > 0:
            self.per_gpu_train_batch_size = self.full_train_batch_size // \
                                            (self.n_gpu * self.world_size * self.gradient_accumulation_steps)
        else:
            self.per_gpu_train_batch_size = self.full_train_batch_size // self.gradient_accumulation_steps

        self.stop_time = None
        if 'TIME_LIMIT_MINS' in os.environ:
            self.stop_time = time.time() + 60 * (int(os.environ['TIME_LIMIT_MINS']) - 5)

    def _post_init(self):
        self._basic_post_init()

        self._setup_logging()

        # Setup distant debugging if needed
        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.server_ip, self.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        logger.warning(
            "On %s, Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            socket.gethostname(),
            self.local_rank,
            self.device,
            self.n_gpu,
            bool(self.local_rank != -1),
            self.fp16,
        )
        logger.info(f'hypers:\n{self}')

    def _setup_logging(self):
        # force our logging style
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if self.log_on_all_nodes:
            grank = self.global_rank
            class HostnameFilter(logging.Filter):
                hostname = socket.gethostname()
                if '.' in hostname:
                    hostname = hostname[0:hostname.find('.')]  # the first part of the hostname

                def filter(self, record):
                    record.hostname = HostnameFilter.hostname
                    record.global_rank = grank
                    return True

            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.addFilter(HostnameFilter())
            format = logging.Formatter('%(hostname)s[%(global_rank)d] %(filename)s:%(lineno)d - %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
            handler.setFormatter(format)
            logging.getLogger('').addHandler(handler)
        else:
            logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
        if self.global_rank != 0 and not self.log_on_all_nodes:
            try:
                logging.getLogger().setLevel(logging.WARNING)
            except:
                pass

    def to_dict(self):
        d = self.__dict__.copy()
        del d['device']
        return d

    def from_dict(self, a_dict):
        fill_from_dict(self, a_dict)
        self._basic_post_init()  # setup device and per_gpu_batch_size
        return self

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def fill_from_args(self):
        fill_from_args(self)
        self._post_init()
        return self
    


class RagHypers(HypersBase):
    def __init__(self):
        super().__init__()
        self.kilt_data = ''
        self.model_name = 'facebook/rag-token-nq'
        self.model_path = ''
        self.no_leading_space = False
        self.n_docs = 5
        self.fold = ''  # {1-n}of{n}
        # only used for train
        self.num_instances = -1
        self.warmup_fraction = 0.1
        

    def get_tokenizer_and_model(self):
        # initialize the model and index
        tokenizer = RagTokenizer.from_pretrained(self.model_name)
        model = RagTokenForGeneration.from_pretrained(self.model_path if self.model_path else self.model_name)
        # model = RagSequenceForGeneration.from_pretrained(self.model_path if self.model_path else self.model_name)
        return tokenizer, model