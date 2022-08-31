import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import def_logger
from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id=-1, model=None, optim=None, tokenizer=None):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    grad_accum_count = args.accum_count if hasattr(args, 'accum_count') else None
    n_gpu = args.world_size if hasattr(args, 'world_size') else 0

    if hasattr(args, 'gpu_ranks') and device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    if hasattr(args, 'model_path'):
        tensorboard_log_dir = args.model_path
        writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
        report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    else:
        report_manager = None
    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager, tokenizer)

    def_logger.info(f'gpu_rank: {gpu_rank}, device_id: {device_id}, trainer: {trainer}, model: {model}')
    if (model):
        n_params = _tally_parameters(model)
        def_logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None, tokenizer=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps if hasattr(args, 'save_checkpoint_steps') else None
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        if grad_accum_count is not None:
            assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        def_logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    def_logger.debug(f'i: {i}, step: {step}, train_steps: {train_steps}, batch: {len(batch)}')

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter, step=0):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                labels = batch.src_sent_labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """

        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval()
        stats = Statistics()

        can_path = '%s_step%d.candidate' % (self.args.result_path, step)
        gold_path = '%s_step%d.gold' % (self.args.result_path, step)
        torch.set_printoptions(precision=10)
        with open(can_path, 'w') as save_pred:
            with open(gold_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        labels = batch.src_sent_labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask_src
                        mask_cls = batch.mask_cls

                        gold = []
                        pred = []

                        if (cal_lead):
                            selected_ids = [list(range(batch.clss.size(1)))] * batch.batch_size
                        elif (cal_oracle):
                            selected_ids = [[j for j in range(batch.clss.size(1)) if labels[i][j] == 1] for i in
                                            range(batch.batch_size)]
                        else:
                            sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)
                            loss = self.loss(sent_scores, labels.float().reshape(sent_scores.shape))
                            loss = (loss * mask.float()).sum()
                            batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                            stats.update(batch_stats)

                            sent_scores = sent_scores + mask.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)

                        def_logger.debug(
                            f'sent_scores.shape: {sent_scores.shape}, sent_scores: {sent_scores}, selected_ids: {selected_ids}')

                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if (len(batch.src_str[i]) == 0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                if (j >= len(batch.src_str[i])):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if (self.args.block_trigram):
                                    if (not _block_tri(candidate, _pred)):
                                        _pred.append(candidate)
                                else:
                                    _pred.append(candidate)

                                if ((not cal_oracle) and (not self.args.recall_eval) and len(_pred) == 3):
                                    break

                            _pred = '<q>'.join(_pred)
                            if (self.args.recall_eval):
                                _pred = ' '.join(_pred.split()[:len(batch.tgt_str[i].split())])

                            pred.append(_pred)
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip() + '\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip() + '\n')
        if (step != -1 and self.args.report_rouge):
            rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
            print('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        self._report_step(0, step, valid_stats=stats)

        return stats

    def predict(self, src_batch, src_str_batch, n_sentence):
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        self.model.eval()

        torch.set_printoptions(precision=10)
        with torch.no_grad():
            segments_ids_batch = []
            cls_ids_batch = []
            mask_batch = []
            max_len = -1
            for src in src_batch:
                if max_len == -1:
                    max_len = len(src)
                else:
                    if max_len != len(src):
                        raise ValueError(f"Max length mismatched! max_len: {max_len}, src_batch: {src_batch}")

            def_logger.debug(f'src_batch: {src_batch}, src_str_batch: {src_str_batch}, n_sentence: {n_sentence}')

            max_cls_len = max([src.count(self.tokenizer.vocab['[CLS]']) for src in src_batch])
            for src in src_batch:
                _segs = [-1] + [i for i, t in enumerate(src) if t == self.tokenizer.vocab['[SEP]']]
                segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
                segments_ids = []
                for i, s in enumerate(segs):
                    if (i % 2 == 0):
                        segments_ids += s * [0]
                    else:
                        segments_ids += s * [1]

                # Pad tokens
                pad = [0] * (max_len - len(segments_ids))
                segments_ids = segments_ids + pad
                def_logger.debug(f'len: {len(segments_ids)}, segments_ids: {segments_ids}')
                segments_ids_batch.append(segments_ids)

                src_wo_pad_len = len(src) - src.count(self.tokenizer.vocab['[PAD]'])
                mask = [1] * src_wo_pad_len + [0] * (max_len - src_wo_pad_len)
                def_logger.debug(f'src_wo_pad_len: {src_wo_pad_len}, len: {len(mask)}, mask: {mask}')
                mask_batch.append(mask)

                cls_ids = [i for i, t in enumerate(src) if t == self.tokenizer.vocab['[CLS]']]
                pad = [-1] * (max_cls_len - len(cls_ids))
                cls_ids = cls_ids + pad
                def_logger.debug(f'len: {len(cls_ids)}, cls_ids: {cls_ids}')
                cls_ids_batch.append(cls_ids)

            # Convert to tensor
            src = torch.tensor(src_batch, dtype=torch.int64, device=self.model.device)
            seg = torch.tensor(segments_ids_batch, dtype=torch.int64, device=self.model.device)
            mask = torch.tensor(mask_batch, dtype=torch.bool, device=self.model.device)
            cls = torch.tensor(cls_ids_batch, dtype=torch.int64, device=self.model.device)
            mask_cls = ~(cls == -1)
            cls[cls == -1] = 0

            # B: batch size, S: max sequence len, N: max sentence len
            # src.shape: torch.Size([B, S]), seg.shape: torch.Size([B, S]), mask.shape: torch.Size([B, S])
            # cls.shape: torch.Size([B, N]), mask_cls.shape: torch.Size([B, N])
            def_logger.debug(
                f'src.shape: {src.shape}, seg.shape: {seg.shape}, mask.shape: {mask.shape}, cls.shape: {cls.shape}, mask_cls.shape: {mask_cls.shape}')

            sent_scores, mask = self.model(src, seg, cls, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores, 1)

            # sent_scores.shape: (B, N), mask.shape: torch.Size([B, N])
            def_logger.debug(
                f'sent_scores.shape: {sent_scores.shape}, sent_scores: {sent_scores}, mask.shape: {mask.shape}, mask: {mask}, selected_ids.shape: {selected_ids.shape}, selected_ids: {selected_ids}')

            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                for j in selected_ids[i][:len(src_str_batch[i])]:
                    if (j >= len(src_str_batch[i])):
                        continue
                    candidate = src_str_batch[i][j].strip()
                    if (self.args.block_trigram):
                        if (not _block_tri(candidate, _pred)):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if (len(_pred) == n_sentence):
                        break
                pred.append(_pred)

            return pred

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.src_sent_labels
            segs = batch.segs
            clss = batch.clss
            mask_src = batch.mask_src
            mask_cls = batch.mask_cls
            def_logger.debug(
                f'src.shape: {src.shape}, labels.shape: {labels.shape}, segs.shape: {segs.shape}, clss.shape: {clss.shape}, mask_src.shape: {mask_src.shape}, mask_cls.shape: {mask_cls.shape}')

            sent_scores, mask = self.model(src, segs, clss, mask_src, mask_cls)
            def_logger.debug(f'sent_scores.shape: {sent_scores.shape}, mask.shape: {mask.shape}')

            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask.float()).sum()
            (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        def_logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
