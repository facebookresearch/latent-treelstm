# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
import random
import argparse
from torch import nn
from utils import get_logger
from functools import partial
from utils import AverageMeter
from utils import EarlyStopping
from utils import get_lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from listops.models import IdealTreeModel
from listops.data_preprocessing import ListOpsDatasetTrees
from listops.data_preprocessing import ListOpsBucketSampler


def make_path_preparations(args):
    # TODO
    seed = 42
    # seed = hash(str(args)) % 1000_000
    ListOpsBucketSampler.random_seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    # logger path
    args_hash = str(hash(str(args)))
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
    logger = get_logger(f"{args.logs_path}/l{args_hash}.log")
    print(f"{args.logs_path}/l{args_hash}.log")
    logger.info(f"args: {str(args)}")
    logger.info(f"args hash: {args_hash}")
    logger.info(f"random seed: {seed}")

    # model path
    args.model_dir = f"{args.model_dir}/m{args_hash}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logger.info(f"checkpoint's dir is: {args.model_dir}")

    # tensorboard path
    tensorboard_path = f"{args.tensorboard_path}/t{args_hash}"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    summary_writer = dict()
    summary_writer["train"] = SummaryWriter(log_dir=os.path.join(tensorboard_path, 'log' + args_hash, 'train'))
    summary_writer["valid"] = SummaryWriter(log_dir=os.path.join(tensorboard_path, 'log' + args_hash, 'valid'))

    return logger, summary_writer


def get_data(args):
    train_dataset = ListOpsDatasetTrees("data/listops/interim/train.tsv", "data/listops/processed/vocab.txt", max_len=130)
    valid_dataset = ListOpsDatasetTrees("data/listops/interim/valid.tsv", "data/listops/processed/vocab.txt", max_len=300)
    test_dataset = ListOpsDatasetTrees("data/listops/interim/test.tsv", "data/listops/processed/vocab.txt")

    train_data_sampler = ListOpsBucketSampler(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                              drop_last=True)
    valid_data_sampler = ListOpsBucketSampler(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
                                              drop_last=False)
    test_data_sampler = ListOpsBucketSampler(dataset=test_dataset, batch_size=args.batch_size//4 + 1, shuffle=False,
                                             drop_last=False)

    train_data = DataLoader(dataset=train_dataset, batch_sampler=train_data_sampler, num_workers=4, pin_memory=True,
                            collate_fn=ListOpsDatasetTrees.collate_fn)
    valid_data = DataLoader(dataset=valid_dataset, batch_sampler=valid_data_sampler, num_workers=4, pin_memory=True,
                            collate_fn=ListOpsDatasetTrees.collate_fn)
    test_data = DataLoader(dataset=test_dataset, batch_sampler=test_data_sampler, num_workers=4, pin_memory=False,
                           collate_fn=ListOpsDatasetTrees.collate_fn)

    # train_data = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                         num_workers=4, pin_memory=True, collate_fn=ListOpsDatasetTrees.collate_fn)
    # valid_data = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True,
    #                         num_workers=4, pin_memory=True, collate_fn=ListOpsDatasetTrees.collate_fn)
    # test_data = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
    #                        num_workers=4, pin_memory=False, collate_fn=ListOpsDatasetTrees.collate_fn)

    args.vocab_size = train_dataset.vocab_size
    args.label_size = train_dataset.label_size
    return train_data, valid_data, test_data


def prepare_optimiser(args, logger, parameters):
    if args.optimizer == "adam":
        optimizer_class = torch.optim.Adam
    elif args.optimizer == "amsgrad":
        optimizer_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.optimizer == "adadelta":
        optimizer_class = torch.optim.Adadelta
    else:
        optimizer_class = torch.optim.SGD
    optimizer = optimizer_class(params=parameters, lr=args.lr, weight_decay=args.l2_weight)
    lr_scheduler = get_lr_scheduler(logger, optimizer, patience=args.lr_scheduler_patience,
                                    threshold=args.lr_scheduler_threshold)
    es = EarlyStopping(mode="max", patience=args.es_patience, threshold=args.es_threshold)
    return optimizer, lr_scheduler, es


def perform_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer.step()
    optimizer.zero_grad()


def test(test_data, model, device, logger):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    model.eval()
    start = time.time()
    with torch.no_grad():
        for labels, tokens, trees, mask in test_data:
            labels = labels.to(device=device, non_blocking=True)
            tokens = tokens.to(device=device, non_blocking=True)
            trees = [e.to(device=device, non_blocking=True) for e in trees]
            mask = mask.to(device=device, non_blocking=True)

            loading_time_meter.update(time.time() - start)

            ce_loss, pred_labels = model(tokens, trees, mask, labels)

            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            n = mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            model.reset_memory_managers()
            batch_time_meter.update(time.time() - start)
            start = time.time()

    logger.info(f"Test: ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")
    logger.info("done")

    return accuracy_meter.avg


def validate(valid_data, model, epoch, device, logger, summary_writer):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    idle_comp_meter = AverageMeter()

    model.eval()
    start = time.time()
    with torch.no_grad():
        for labels, tokens, trees, mask in valid_data:
            labels = labels.to(device=device, non_blocking=True)
            tokens = tokens.to(device=device, non_blocking=True)
            trees = [e.to(device=device, non_blocking=True) for e in trees]
            mask = mask.to(device=device, non_blocking=True)
            loading_time_meter.update(time.time() - start)

            ce_loss, pred_labels = model(tokens, trees, mask, labels)

            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            n = mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            idle_comp_meter.update((mask.sum() / mask.numel()).item())
            model.reset_memory_managers()
            batch_time_meter.update(time.time() - start)
            start = time.time()

    logger.info(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"idle_comp: {idle_comp_meter.avg} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")

    summary_writer["valid"].add_scalar(tag="ce", scalar_value=ce_loss_meter.avg, global_step=global_step)
    summary_writer["valid"].add_scalar(tag="accuracy", scalar_value=accuracy_meter.avg, global_step=global_step)
    summary_writer["valid"].add_scalar(tag="idle_comp", scalar_value=idle_comp_meter.avg, global_step=global_step)

    model.train()
    return accuracy_meter.avg


def train(train_data, valid_data, model, optimizer, lr_scheduler, es, epoch, args, logger, summary_writer):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    idle_comp_meter = AverageMeter()

    device = args.gpu_id
    model.train()
    start = time.time()
    for batch_idx, (labels, tokens, trees, mask) in enumerate(train_data):
        labels = labels.to(device=device, non_blocking=True)
        tokens = tokens.to(device=device, non_blocking=True)
        trees = [e.to(device=device, non_blocking=True) for e in trees]
        mask = mask.to(device=device, non_blocking=True)
        loading_time_meter.update(time.time() - start)

        ce_loss, pred_labels = model(tokens, trees, mask, labels)
        ce_loss.backward()
        perform_optimizer_step(optimizer, model, args)

        n = mask.shape[0]
        accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
        accuracy_meter.update(accuracy.item(), n)
        ce_loss_meter.update(ce_loss.item(), n)
        idle_comp_meter.update((mask.sum() / mask.numel()).item())
        model.reset_memory_managers()
        batch_time_meter.update(time.time() - start)

        global global_step
        summary_writer["train"].add_scalar(tag="ce", scalar_value=ce_loss.item(), global_step=global_step)
        summary_writer["train"].add_scalar(tag="accuracy", scalar_value=accuracy.item(), global_step=global_step)
        summary_writer["train"].add_scalar(tag="idle_comp", scalar_value=idle_comp_meter.value, global_step=global_step)
        global_step += 1

        if (batch_idx + 1) % (len(train_data) // 3) == 0:
            logger.info(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
                        f"accuracy: {accuracy_meter.avg:.4f} idle_comp: {idle_comp_meter.avg} "
                        f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")
            val_accuracy = validate(valid_data, model, epoch, device, logger, summary_writer)
            lr_scheduler.step(val_accuracy)
            es.step(val_accuracy)
            global best_model_path
            if es.is_converged:
                return
            if es.is_improved():
                logger.info("saving model...")
                best_model_path = f"{args.model_dir}/{epoch}-{batch_idx}.mdl"
                torch.save({"epoch": epoch, "batch_idx": batch_idx, "state_dict": model.state_dict()}, best_model_path)
            model.train()
        start = time.time()


def main(args):
    logger, summary_writer = make_path_preparations(args)
    train_data, valid_data, test_data = get_data(args)
    model = IdealTreeModel(vocab_size=args.vocab_size,
                           word_dim=args.word_dim,
                           hidden_dim=args.hidden_dim,
                           label_dim=args.label_size,
                           leaf_transformation=args.leaf_transformation,
                           trans_hidden_dim=args.trans_hidden_dim).cuda(args.gpu_id)
    optimizer, lr_scheduler, es = prepare_optimiser(args, logger, model.parameters())

    validate(valid_data, model, 0, args.gpu_id, logger, summary_writer)
    for epoch in range(args.max_epoch):
        train(train_data, valid_data, model, optimizer, lr_scheduler, es, epoch, args, logger, summary_writer)
        if es.is_converged:
            break
    print(best_model_path)
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    test(test_data, model, args.gpu_id, logger)


if __name__ == "__main__":
    args = {"word-dim":                   128,
            "hidden-dim":                 128,
            "leaf-transformation":        "no_transformation",
            "trans-hidden_dim":           128,
            "clip-grad-norm":             0.5,
            # "optimizer":                  "adadelta",
            # "lr":                         1.0,
            "optimizer":                  "amsgrad",
            "lr":                         0.001,
            "lr-scheduler-patience":      8,
            "lr-scheduler-threshold":     0.005,
            "l2-weight":                  0.0001,
            "batch-size":                 256,
            "max-epoch":                  300,
            "es-patience":                20,
            "es-threshold":               0.005,
            "gpu-id":                     1,
            "model-dir":                  "data/listops/tree_lstm/models/exp0",
            "logs-path":                  "data/listops/tree_lstm/logs/exp0",
            "tensorboard-path":           "data/listops/tree_lstm/tensorboard/exp0"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--word-dim", required=False, default=args["word-dim"], type=int)
    parser.add_argument("--hidden-dim", required=False, default=args["hidden-dim"], type=int)
    parser.add_argument("--leaf-transformation", required=False, default=args["leaf-transformation"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--trans-hidden_dim", required=False, default=args["trans-hidden_dim"], type=int)

    parser.add_argument("--clip-grad-norm", default=args["clip-grad-norm"], type=float,
                        help="If the value is less or equal to zero clipping is not performed.")
    parser.add_argument("--optimizer", required=False, default=args["optimizer"], choices=["adam", "sgd", "adadelta", "amsgrad"])
    parser.add_argument("--lr", required=False, default=args["lr"], type=float)
    parser.add_argument("--lr-scheduler-patience", required=False, default=args["lr-scheduler-patience"], type=int)
    parser.add_argument("--lr-scheduler-threshold", required=False, default=args["lr-scheduler-threshold"], type=float)
    parser.add_argument("--l2-weight", required=False, default=args["l2-weight"], type=float)
    parser.add_argument("--batch-size", required=False, default=args["batch-size"], type=int)

    parser.add_argument("--max-epoch", required=False, default=args["max-epoch"], type=int)
    parser.add_argument("--es-patience", required=False, default=args["es-patience"], type=int)
    parser.add_argument("--es-threshold", required=False, default=args["es-threshold"], type=float)
    parser.add_argument("--gpu-id", required=False, default=args["gpu-id"], type=int)
    parser.add_argument("--model-dir", required=False, default=args["model-dir"], type=str)
    parser.add_argument("--logs-path", required=False, default=args["logs-path"], type=str)
    parser.add_argument("--tensorboard-path", required=False, default=args["tensorboard-path"], type=str)

    global_step = 0
    best_model_path = None
    args = parser.parse_args()
    with torch.cuda.device(args.gpu_id):
        main(args)
