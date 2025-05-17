import os
from logging import getLogger
import copy
import Levenshtein

import torch
import torch.nn.functional as F
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from dual_arm_cluster_tool.concurrent_flow.envs.ddcfEnv import ddcfEnv as Env
from dual_arm_cluster_tool.concurrent_flow.envs.ddcfEnv import State

from model.model_concat import CONCATNet as CONCATModel

from utils.utils import (get_result_folder, LogData, TimeEstimator,
                         util_save_log_image_with_label, util_print_log_array,
                         AverageMeter, batchify, unbatchify, batchify_dataclass, unbatchify_dataclass)



class NCTSTrainer:
    def __init__(self, env_params, model_params,
                 optimizer_params, trainer_params):
        # save arguments
        # --------------------------------------------------------------------------------------------
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        # --------------------------------------------------------------------------------------------
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        # --------------------------------------------------------------------------------------------
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            #torch.set_default_tensor_type('torch.FloatTensor')
        self.model_params['device'] = self.device

        # Main Components
        # --------------------------------------------------------------------------------------------
        #self.envs = [Env(**self.env_params) for _ in range(self.trainer_params['train_batch_size'])]
        self.model = CONCATModel(**self.env_params, **self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # restore
        # --------------------------------------------------------------------------------------------
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            #checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint_file = next((f for f in os.listdir(model_load['path']) if f.startswith('checkpoint-')), None)
            checkpoint_fullname = model_load['path'] +'/' + checkpoint_file
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not model_load['load_model_only']:
                self.start_epoch = 1 + model_load['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.last_epoch = model_load['epoch']-1

            self.logger.info('######################################')
            self.logger.info(f'Saved Model Loaded !! >>>>> {checkpoint_fullname}')
            self.logger.info('######################################')

        # utility
        # --------------------------------------------------------------------------------------------
        self.time_estimator = TimeEstimator()
        self.grad_accumulate_cnt = 0 # for gradient accumulation


    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            # -----------------------------------------------------------------------------------------
            self.scheduler.step()

            # Train
            # -----------------------------------------------------------------------------------------
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # Logs
            # -----------------------------------------------------------------------------------------
            elapsed_time_str, remain_time_str =\
                    self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])

            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            # Save, checkpoint
            # -----------------------------------------------------------------------------------------
            saving = True
            if saving == True:
                all_done = (epoch == self.trainer_params['epochs'])
                model_save_interval = self.trainer_params['logging']['model_save_interval']
                img_save_interval = self.trainer_params['logging']['img_save_interval']

                if epoch > 1:  # save latest images, every epoch
                    self.logger.info("Saving log_image")
                    image_prefix = '{}/latest'.format(self.result_folder)
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                self.result_log, labels=['train_score'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                self.result_log, labels=['train_loss'])

                if all_done or (epoch % model_save_interval) == 0:
                    self.logger.info("Saving trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }
                    torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

                if all_done or (epoch % img_save_interval) == 0:
                    image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                self.result_log, labels=['train_score'])
                    util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                self.result_log, labels=['train_loss'])

                if all_done:
                    self.logger.info(" *** Training Done *** ")
                    self.logger.info("Now, printing log array...")
                    util_print_log_array(self.logger, self.result_log)

            all_done = (epoch == self.trainer_params['epochs'])
            if all_done:
                self.logger.info(" *** Training Done !!!*** ")

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        base_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        self.pomo_size = self.trainer_params['pomo_size']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            # set minibatch size
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # train one batch
            avg_score, avg_loss = self._train_one_batch(batch_size)

            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            # update counter
            episode += batch_size
            loop_cnt += 1

            # Log First 10 Batch, only  at the first epoch
            if epoch == self.start_epoch:
                if loop_cnt <= 20:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}, Loss: {:.4f}'
                        .format(epoch, episode, train_num_episode,
                                100. * episode / train_num_episode,
                                score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        def _stack_states(states: list):
            return State(**{field: torch.stack([getattr(state, field) for state in states])
                            for field in State.__dataclass_fields__})

        # Encoding embedding vector
        # ---------------------------------------------------------------------
        self.model.train()
        states = []
        envs = []
        for b in range(batch_size):
            env = Env(**self.env_params)
            state = env.reset()

            envs.append(env)
            states.append(state)

        state = _stack_states(states)
        state.to(self.device)
        self.model.encoding(state)

        # batchify multistart(POMO) or single start
        # ---------------------------------------------------------------------
        if self.pomo_size != 1:
            # state batchify
            state = batchify_dataclass(state, self.pomo_size)
            state.batch_idx = torch.arange(state.batch_size(), device=self.device) # batch_idx post processing

            # model batchify
            self.model.row_embed = batchify(self.model.row_embed, self.pomo_size)
            self.model.col_embed = batchify(self.model.col_embed, self.pomo_size)

            # env batchify
            envs = [copy.deepcopy(env) for _ in range(self.pomo_size) for env in envs]

        # Rollout(deocding)
        # ---------------------------------------------------------------------
        outputs = []
        actions = []
        makespan = [1e10 for _ in range(state.batch_size())]

        while not state.done.all():
            action, prob = self.model(state)
            # Check GPU memory usage
            if torch.cuda.memory_allocated(self.device) > 16000000000:
                self.logger.warning("GPU memory usage exceeded the limit. Breaking the rollout loop.")

            states = []
            for b, a in enumerate(action):
                state = envs[b].step(a.item())
                states.append(state)
                if envs[b].done and makespan[b] == 1e10:
                    makespan[b] = copy.deepcopy(envs[b].clock)

            state = _stack_states(states)
            state.batch_idx = torch.arange(state.batch_size())
            state.to(self.device)

            outputs.append(prob)
            actions.append(action)

        #throughput = torch.tensor([state.clock for state in states], device=self.device)
        quantity = envs[0].done_quantity * envs[0].foup_size
        state.reward = -torch.tensor(makespan, device=self.device) / quantity

        # Learning
        # ---------------------------------------------------------------------
        # stack outputs, actions
        outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        #lots = torch.stack(lots, 1)

        if self.pomo_size != 1:
            # unbatchify
            reward = unbatchify(state.reward, self.pomo_size)
            actions = unbatchify(actions, self.pomo_size)
            outputs = unbatchify(outputs, self.pomo_size)
            #lots = unbatchify(lots, self.pomo_size)
        else:
            reward = state.reward

        # calculate loss
        if self.trainer_params['pomo_type'] == 'avg':
            baseline = reward.float().mean(dim=1, keepdims=True)
            advantage = reward - baseline                                   # (batch, pomo)
            log_prob = outputs.log().sum(dim=2)                             # (batch, pomo)
            loss = -advantage * log_prob                                    # (batch, pomo) Minus sign: to increase reward
            loss_mean = loss.mean()

        elif self.trainer_params['pomo_type'] == 'max':
            alpha = self.trainer_params['pomo_alpha']
            baseline = reward.float().mean(dim=1, keepdims=True)
            # weighted advantage value
            advantage =(reward - baseline)/alpha                                   # (batch, pomo)
            _, best_traj = advantage.max(dim=1)
            advantage[torch.arange(batch_size), best_traj] = advantage[torch.arange(batch_size), best_traj] * alpha

            log_prob = outputs.log().sum(dim=2)                             # (batch, pomo)
            loss = -advantage * log_prob                                    # (batch, pomo) Minus sign: to increase reward
            loss_mean = loss.mean()

        elif self.trainer_params['pomo_type'] == 'elite':
            alpha = self.trainer_params['pomo_alpha']
            elite_ratio = self.trainer_params['pomo_elite_ratio']
            num_traj = reward.size(-1)
            num_k = int(num_traj * elite_ratio)
            baseline = reward.float().mean(dim=1, keepdims=True)
            advantage = (reward - baseline) / alpha
            _, top_traj = torch.topk(advantage, num_k, dim=1)
            for i in range(num_k):
                advantage[torch.arange(batch_size), top_traj[:, i]] =\
                      advantage[torch.arange(batch_size), top_traj[:, i]] * (num_k-i)

            log_prob = outputs.log().sum(dim=2)
            loss = -advantage *log_prob
            loss_mean = loss.mean()

        elif self.trainer_params['pomo_type'] == 'novel_elite':
            novelty_metric = self.trainer_params.get('similarity_metric', 'cosine') # cosine, Levenshtein
            alpha = self.trainer_params['pomo_alpha']
            novelty_weight = self.trainer_params['novelty_weight']
            num_k = int(reward.size(-1) / 10)
            baseline = reward.float().mean(dim=1, keepdims=True)

            if novelty_metric == 'cosine':
                """
                process_time = unbatchify(self.env.recipe_table['process_time'], self.pomo_size)
                process_time_base_sequence = []
                for stage in range(1, self.env.loc.num_stage+1):
                    for lot in range(lots.size(-1)):
                        process_time_base_sequence.append(
                            gather_by_index(process_time[:, :, :, stage].to(lots.device), lots[:, :, lot], dim=-1)
                            )
                process_time_base_sequence = torch.stack(process_time_base_sequence, dim=-1)

                # get similarity
                seq1 = process_time_base_sequence.unsqueeze(2)
                seq2 = process_time_base_sequence.unsqueeze(1)
                cosine_similarities = F.cosine_similarity(seq1, seq2, dim=-1)
                mask = torch.eye(self.pomo_size).bool().to(process_time_base_sequence.device).unsqueeze(0)
                cosine_similarities.masked_fill_(mask, 0)
                avg_distance = -cosine_similarities.sum(-1)

                novelty_score = novelty_weight*\
                    (avg_distance - avg_distance.min())/(avg_distance.max() - avg_distance.min())
                """
                NotImplementedError

            elif novelty_metric == 'levenshtein':
                pomo_size = reward.size(1)
                distances = torch.zeros((batch_size, pomo_size , pomo_size)).to(self.env.device)
                for batch in range(batch_size):
                    for i in range(pomo_size):
                        for j in range(pomo_size):
                            if i!=j:
                                seq1 = ''.join(map(str, actions[batch, i].tolist()))
                                seq2 = ''.join(map(str, actions[batch, j].tolist()))
                                distances[batch, i, j] = Levenshtein.distance(seq1, seq2)
                avg_distance = distances.sum(-1)
                novelty_score = novelty_weight *\
                    (avg_distance - avg_distance.min())/(avg_distance.max()-avg_distance.min())

            _, top_traj = torch.topk(-reward+novelty_score, num_k, dim=1)

            advantage = (reward - baseline) / alpha
            for i in range(num_k):
                advantage[torch.arange(batch_size), top_traj[:, i]] =\
                      advantage[torch.arange(batch_size), top_traj[:, i]] * (num_k-i)
            log_prob = outputs.log().sum(dim=2)
            loss = -advantage *log_prob
            loss_mean = loss.mean()

        # score
        max_pomo_reward, _ = reward.max(dim=1)                          # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()                    # negative sign to make positive value

        # Step
        loss_mean = loss_mean / self.trainer_params['iters_to_accumulate']
        loss_mean.backward()
        if (self.grad_accumulate_cnt+1) % self.trainer_params['iters_to_accumulate'] == 0:
            # Wait for several backward steps
            self.optimizer.step()
            self.model.zero_grad()
            self.optimizer.step()
            self.grad_accumulate_cnt = 0
        else:
            self.grad_accumulate_cnt += 1

        return score_mean.item(), loss_mean.item()

