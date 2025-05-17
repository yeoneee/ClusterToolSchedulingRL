"""
The MIT License

Copyright (c) 2024 NCTS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0



##########################################################################################
# Path Config

import os
import sys

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
#sys.path.insert(0, "..")  # for problem_def
#sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
import argparse
from utils.utils import create_logger, copy_all_src
from trainer import NCTSTrainer as Trainer
from datetime import datetime


##########################################################################################


parser = argparse.ArgumentParser(description='Train a model with a specific number of lot types')
parser.add_argument('--foup_size', type=int, default=25, help='Size of the foup')
parser.add_argument('--group1_stage', type=int, default=0, help='Stages for type 1')
parser.add_argument('--group1_min_prs_time', type=int, default=10, help='Minimum processing time for type 1')
parser.add_argument('--group1_max_prs_time', type=int, default=300, help='Maximum processing time for type 1')
parser.add_argument('--group2_stage', type=int, default=1, help='Stages for type 2')
parser.add_argument('--group2_min_prs_time', type=int, default=10, help='Minimum processing time for type 2')
parser.add_argument('--group2_max_prs_time', type=int, default=300, help='Maximum processing time for type 2')
parser.add_argument('--prod_quantity', type=int, default=10, help='Production quantity (Unit: FOUP)')
parser.add_argument('--done_quantity', type=int, default=15, help='Done Production quantity (Unit: Wafer)')
parser.add_argument('--num_lot_type', type=int, default=2, help='Total number of lot types')

parser.add_argument('--model_type', type=str, default='concat', help='Model type = {is, rms, concat}')
parser.add_argument('--input_action', type=str, default='wafer', help='loadlock input action type = {wafer, type}')
parser.add_argument('--epoch', type=int, default=15, help='Number of train epochs')
parser.add_argument('--train_episodes', type=int, default=1*1000, help='Number of train episodes')
parser.add_argument('--train_batch_size', type=int, default=20, help='Number of batch size')
parser.add_argument('--iters_to_accumulate', type=int, default=4, help='Number of iters to accumulate')

parser.add_argument('--pomo_size', type=int, default=20, help='Number of multi runs')
parser.add_argument('--pomo_type', type=str, default='max', help='POMO type = {avg, max, elite}')
parser.add_argument('--pomo_alpha', type=int, default=10, help='POMO type')
parser.add_argument('--pomo_elite_ratio', type=int, default=0.1, help='POMO elite ratio')
parser.add_argument('--novelty_weight', type=float, default=0.1, help='POMO type')


args = parser.parse_args()

stage_list = [
    [1,1],
    [1,1,1],
]

# parameters
env_params = {
    'foup_size': args.foup_size,
    'group1_stage': stage_list[args.group1_stage],
    'group1_min_prs_time': args.group1_min_prs_time,
    'group1_max_prs_time': args.group1_max_prs_time,
    'group2_stage': stage_list[args.group2_stage],
    'group2_min_prs_time': args.group2_min_prs_time,
    'group2_max_prs_time': args.group2_max_prs_time,
    'prod_quantity': args.prod_quantity,
    'done_quantity': args.done_quantity,
    'num_lot_type': args.num_lot_type,
}

model_params = {
    'type': args.model_type,
    'input_action': args.input_action,
    'purge': False,
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 3,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'argmax',
    'normalize': 'instance' if env_params['num_lot_type'] > 1 else 'batch',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [101, 201],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'pomo_size': args.pomo_size,
    'pomo_type': args.pomo_type,
    'pomo_alpha': args.pomo_alpha,
    'pomo_elite_ratio': args.pomo_elite_ratio,
    'novelty_weight': args.novelty_weight,
    'epochs': args.epoch,
    'train_episodes': args.train_episodes,
    'iters_to_accumulate': args.iters_to_accumulate,
    'train_batch_size': args.train_batch_size,
    'logging': {
        'model_save_interval': 5,
        'img_save_interval': 50,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': f'./experiments/is/purge/',
        'epoch': 10,  # epoch version of pre-trained model to load.
        'load_model_only': True, # load only model, not optimizer
        'sub_path': f'./experiments/rms/',
        'sub_epoch': 15,
    }
}

current_time = datetime.now().strftime("%m%d_%H%M")
desc = f'({args.group1_stage},{args.group2_stage})/[{args.group1_min_prs_time}_{args.group1_max_prs_time}]_[{args.group2_min_prs_time}_{args.group2_max_prs_time}]/{current_time}'
logger_params = {
    'log_file': {
        'desc': desc,
        'filename': 'log.txt'
    }
}

##########################################################################################
# main
def main():

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

    if DEBUG_MODE:
        # Print Scehdule for last batch problem
        # env.print_schedule()
        pass

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

"======================= Seed fix =================================="
import torch
import random
import numpy as np

SEED = 1000
torch.backends.cudnn.deterministic = True
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == "__main__":
    main()
