import logging
import os
import torch
from torch.nn import functional as F
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from squad_fine_tuning.set_seed_and_dist import set_seed, N_GPU, DEVICE
from logger import logger
from distilbert_run_and_hpo_configurations.distilbert_squad_hpo_parameters import SGDHyperparameter, DistillationHyperparameter
from distilbert_run_and_hpo_configurations.distilbert_squad_run_parameters import RunParameters
from squad_fine_tuning.eval_squad_distillation import EvalSquadDistillation
from distilbert_data_model_loaders.load_squad_dataset import load_and_cache_examples

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)


class TrainSquadDistillation(object):

    def __init__(self, model_class, tokenizer, tokenizer_class, teacher):
        self.model_class = model_class
        self.tokenizer = tokenizer
        self.tokenizer_class = tokenizer_class
        self.teacher = teacher

    def train(self, model, args_dict, train_dataset, sigopt_run):
        """ Train the model """
        logging.debug("model architecture being used: {}".format(model))

        train_batch_size, train_dataloader = self.get_train_dataloader(args_dict, train_dataset)
        t_total = self.get_num_optimization_steps(args_dict, train_dataloader)
        logging.debug("Number of optimization steps: %d. If equal to 0, training will error out.", t_total)

        optimizer, scheduler = self.get_optimizer_scheduler(model, args_dict, t_total)

        if args_dict[RunParameters.FP_16.value] is True:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args_dict["fp16_opt_level"])
        else:
            model = model

        # multi-gpu training (should be after apex fp16 initialization)
        if args_dict[N_GPU] > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args_dict[RunParameters.LOCAL_RANK.value] != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args_dict[RunParameters.LOCAL_RANK.value]],
                output_device=args_dict[RunParameters.LOCAL_RANK.value])

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args_dict[RunParameters.NUM_TRAIN_EPOCHS.value])
        logger.info("  Instantaneous batch size per GPU = %d", args_dict[
            SGDHyperparameter.PER_COMPUTE_TRAIN_BATCH_SIZE.value])
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args_dict[train_batch_size]
            * args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value]
            * (torch.distributed.get_world_size() if args_dict[RunParameters.LOCAL_RANK.value] != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d",
                    args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value])
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args_dict[RunParameters.MODEL_NAME_OR_PATH.value]):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args_dict[RunParameters.MODEL_NAME_OR_PATH.value].split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value])
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value])

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args_dict[RunParameters.NUM_TRAIN_EPOCHS.value]), desc="Epoch",
            disable=args_dict[RunParameters.LOCAL_RANK.value] not in [-1, 0]
        )
        # Added here for reproductibility
        set_seed(args_dict)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                                  disable=args_dict[RunParameters.LOCAL_RANK.value] not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                loss = self.forward_pass(args_dict, batch, model)
                self.backward_pass(args_dict, loss, optimizer)
                tr_loss += loss.item()

                if (step + 1) % args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value] == 0:
                    if args_dict[RunParameters.FP_16.value] is True:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                       args_dict[SGDHyperparameter.MAX_GRAD_NORM.value])
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       args_dict[SGDHyperparameter.MAX_GRAD_NORM.value])

                    logging.info("stepping optimzer, scheduler, and global step")

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if args_dict[RunParameters.LOCAL_RANK.value] in [-1, 0] and args_dict[
                        RunParameters.LOGGING_STEPS.value] > 0 and global_step % args_dict[
                        RunParameters.LOGGING_STEPS.value] == 0:
                        logging.info("logging metrics for global step {}".format(global_step))
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = dict()
                        if args_dict[RunParameters.LOCAL_RANK.value] == -1 and args_dict[
                            RunParameters.EVALUATE_DURING_TRAINING.value]:
                            results, eval_time = self.evaluate(args_dict, model, global_step)
                            logging.info("validation run results: {}".format(results))

                        current_lr = scheduler.get_lr()[0]
                        current_loss = (tr_loss - logging_loss) / args_dict[RunParameters.LOGGING_STEPS.value]
                        logging_loss = tr_loss
                        if sigopt_run is not None:
                            logging.info("logging SigOpt checkpoint for current global step: {}".format(global_step))
                            self.sigopt_runs_update(sigopt_run, current_loss, current_lr, results)

                    if args_dict[RunParameters.LOCAL_RANK.value] in [-1, 0] and args_dict[
                        RunParameters.SAVE_STEPS.value] > 0 and global_step % args_dict[
                        RunParameters.SAVE_STEPS.value] == 0:
                        # Save model checkpoint
                        self.checkpoint(args_dict, model, optimizer, scheduler)

                if args_dict[RunParameters.MAX_STEPS.value] > 0 and global_step > args_dict[
                    RunParameters.MAX_STEPS.value]:
                    epoch_iterator.close()
                    break
            if args_dict[RunParameters.MAX_STEPS.value] > 0 and global_step > args_dict[RunParameters.MAX_STEPS.value]:
                train_iterator.close()
                break

        return model, global_step, tr_loss / global_step

    def backward_pass(self, args_dict, loss, optimizer):
        if args_dict[RunParameters.FP_16.value] is True:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def forward_pass(self, args_dict, batch, model):
        if self.teacher is not None:
            self.teacher.eval()
        batch = tuple(t.to(args_dict[DEVICE]) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        # forward pass - loss calculation
        outputs = model(**inputs)
        loss, start_logits_stu, end_logits_stu = outputs
        # Distillation loss
        if self.teacher is not None:
            loss = self.calc_distillation_loss(args_dict, batch, end_logits_stu, inputs, loss, start_logits_stu)
        if args_dict[N_GPU] > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value] > 1:
            loss = loss / args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value]
        return loss

    def evaluate(self, args_dict, model, global_step):
        dataset, examples, features = load_and_cache_examples(args_dict=args_dict, tokenizer=self.tokenizer,
                                                            evaluate=True, output_examples=True)
        eval_squad = EvalSquadDistillation(args_dict=args_dict, tokenizer=self.tokenizer, global_step=global_step)
        results, eval_time = eval_squad.evaluate(model=model, dataset=dataset, examples=examples, features=features)
        return results, eval_time

    def sigopt_runs_update(self, sigopt_run, current_loss, current_lr, results):
        # update Sigopt Run
        sigopt_checkpoint_metrics = dict()
        sigopt_checkpoint_metrics.update(results)
        sigopt_checkpoint_metrics["lr"] = current_lr
        sigopt_checkpoint_metrics["loss"] = current_loss
        sigopt_run.log_checkpoint(sigopt_checkpoint_metrics)

    def checkpoint(self, args_dict, model, optimizer, scheduler):
        model_save_dir = args_dict[RunParameters.OUTPUT_DIR.value]
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(model_save_dir)
        self.tokenizer.save_pretrained(model_save_dir)
        torch.save(args_dict, os.path.join(model_save_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", model_save_dir)
        torch.save(optimizer.state_dict(), os.path.join(model_save_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(model_save_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", model_save_dir)

    def calc_distillation_loss(self, args_dict, batch, end_logits_stu, inputs, loss, start_logits_stu):
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = None if args_dict[RunParameters.TEACHER_TYPE.value] == "xlm" else \
                batch[
                    2]
        with torch.no_grad():
            start_logits_tea, end_logits_tea = self.teacher(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
            )
        assert start_logits_tea.size() == start_logits_stu.size()
        assert end_logits_tea.size() == end_logits_stu.size()
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        loss_start = loss_fct(
            F.log_softmax(start_logits_stu / args_dict[DistillationHyperparameter.TEMPERATURE.value],
                          dim=-1),
            F.softmax(start_logits_tea / args_dict[DistillationHyperparameter.TEMPERATURE.value], dim=-1),
        ) * (args_dict[DistillationHyperparameter.TEMPERATURE.value] ** 2)
        loss_end = loss_fct(
            F.log_softmax(end_logits_stu / args_dict[DistillationHyperparameter.TEMPERATURE.value],
                          dim=-1),
            F.softmax(end_logits_tea / args_dict[DistillationHyperparameter.TEMPERATURE.value], dim=-1),
        ) * (args_dict[DistillationHyperparameter.TEMPERATURE.value] ** 2)
        loss_ce = (loss_start + loss_end) / 2.0
        loss = args_dict[DistillationHyperparameter.ALPHA_CE.value] * loss_ce + args_dict[
            DistillationHyperparameter.ALPHA_SQUAD.value] * loss
        return loss

    def get_num_optimization_steps(self, args_dict, train_dataloader):
        if args_dict[RunParameters.MAX_STEPS.value] > 0:
            t_total = args_dict[RunParameters.MAX_STEPS.value]
            args_dict[RunParameters.NUM_TRAIN_EPOCHS.value] = args_dict[RunParameters.MAX_STEPS.value] // (
                len(train_dataloader) // args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value]) + 1
        else:
            t_total = len(train_dataloader) // args_dict[SGDHyperparameter.GRADIENT_ACCUMULATION_STEPS.value] * \
                      args_dict[RunParameters.NUM_TRAIN_EPOCHS.value]
        return t_total

    def get_optimizer_scheduler(self, model, args_dict, t_total):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args_dict[SGDHyperparameter.WEIGHT_DECAY.value],
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args_dict[SGDHyperparameter.LEARNING_RATE.value],
                          eps=args_dict[SGDHyperparameter.ADAM_EPSILON.value])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args_dict[SGDHyperparameter.WARM_UP_STEPS.value], num_training_steps=t_total
        )
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(
            os.path.join(args_dict[RunParameters.MODEL_NAME_OR_PATH.value], "optimizer.pt")) and os.path.isfile(
            os.path.join(args_dict[RunParameters.MODEL_NAME_OR_PATH.value], "scheduler.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args_dict[RunParameters.MODEL_NAME_OR_PATH.value],
                                                              "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args_dict[RunParameters.MODEL_NAME_OR_PATH.value],
                                                              "scheduler.pt")))
        return optimizer, scheduler

    def get_train_dataloader(self, args_dict, train_dataset):
        train_batch_size = "train_batch_size"
        args_dict[train_batch_size] = args_dict[SGDHyperparameter.PER_COMPUTE_TRAIN_BATCH_SIZE.value] * max(1,
                                                                                                            args_dict[
                                                                                                                N_GPU])
        train_sampler = RandomSampler(train_dataset) if args_dict[
                                                            RunParameters.LOCAL_RANK.value] == -1 else DistributedSampler(
            train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args_dict[train_batch_size])
        return train_batch_size, train_dataloader
