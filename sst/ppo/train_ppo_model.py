# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
import random
import argparse
import torchtext
from torch import nn
from utils import get_logger
from functools import partial
from utils import AverageMeter
from sst.models import PpoModel
from utils import EarlyStopping
from utils import length_to_mask
from utils import get_lr_scheduler
from tensorboardX import SummaryWriter


def make_path_preparations(args):
    seed = hash(str(args)) % 1000_000
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


def get_data(args, logger):
    text_field = torchtext.data.Field(lower=args.lower, include_lengths=True, batch_first=True)
    label_field = torchtext.data.Field(sequential=False, unk_token=None)
    dataset_splits = torchtext.datasets.SST.splits(
        root="data", text_field=text_field, label_field=label_field,
        fine_grained=args.fine_grained, train_subtrees=True,
        filter_pred=None if args.fine_grained else lambda e: (e.label != 'neutral' and len(e.text) > 1))
    text_field.build_vocab(*dataset_splits, vectors=args.vectors, vectors_cache="data/vectors_cache")
    label_field.build_vocab(*dataset_splits)

    logger.info(f'Pretrained vectors: {args.vectors}')
    logger.info(f'Number of classes: {len(label_field.vocab)}')

    train_data, valid_data, test_data = torchtext.data.BucketIterator.splits(datasets=dataset_splits,
                                                                             batch_size=args.batch_size,
                                                                             device=torch.device(args.gpu_id),
                                                                             repeat=False)
    print('train', len(train_data))
    print('valid', len(valid_data))

    args.vocab_size = text_field.vocab.vectors.shape[1]
    args.label_size = len(label_field.vocab)
    vectors = text_field.vocab.vectors if args.vectors else None

    return train_data, valid_data, test_data, vectors


def prepare_optimisers(args, logger, policy_parameters, environment_parameters):
    if args.env_optimizer == "adam":
        env_opt_class = torch.optim.Adam
    elif args.env_optimizer == "amsgrad":
        env_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.env_optimizer == "adadelta":
        env_opt_class = torch.optim.Adadelta
    else:
        env_opt_class = torch.optim.SGD

    if args.pol_optimizer == "adam":
        pol_opt_class = torch.optim.Adam
    elif args.pol_optimizer == "amsgrad":
        pol_opt_class = partial(torch.optim.Adam, amsgrad=True)
    elif args.pol_optimizer == "adadelta":
        pol_opt_class = torch.optim.Adadelta
    else:
        pol_opt_class = torch.optim.SGD

    optimizer = {"policy": pol_opt_class(params=policy_parameters, lr=args.pol_lr, weight_decay=args.l2_weight),
                 "env": env_opt_class(params=environment_parameters, lr=args.env_lr, weight_decay=args.l2_weight)}
    lr_scheduler = {"policy": get_lr_scheduler(logger, optimizer["policy"], patience=args.lr_scheduler_patience,
                                               threshold=args.lr_scheduler_threshold),
                    "env": get_lr_scheduler(logger, optimizer["env"], patience=args.lr_scheduler_patience,
                                            threshold=args.lr_scheduler_threshold)}
    es = EarlyStopping(mode="max", patience=args.es_patience, threshold=args.es_threshold)
    return optimizer, lr_scheduler, es


def perform_env_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.get_environment_parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["env"].step()
    optimizer["env"].zero_grad()


def perform_policy_optimizer_step(optimizer, model, args):
    if args.clip_grad_norm > 0:
        nn.utils.clip_grad_norm_(parameters=model.get_policy_parameters(),
                                 max_norm=args.clip_grad_norm,
                                 norm_type=float("inf"))
    optimizer["policy"].step()
    optimizer["policy"].zero_grad()


def test(test_data, model, device, logger):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in test_data:
            tokens, length = batch.text
            labels = batch.label
            mask = length_to_mask(length)
            loading_time_meter.update(time.time() - start)

            pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = \
                model(tokens, mask, labels)
            entropy = entropy.mean()
            normalized_entropy = normalized_entropy.mean()

            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            n = mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            entropy_meter.update(entropy.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            batch_time_meter.update(time.time() - start)
            start = time.time()

    logger.info(f"Test: ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"entropy: {entropy_meter.avg:.4f} n_entropy: {n_entropy_meter.avg:.4f} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")
    logger.info("done")

    return accuracy_meter.avg


def validate(valid_data, model, epoch, device, logger, summary_writer):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()

    model.eval()
    start = time.time()
    with torch.no_grad():
        for batch in valid_data:
            tokens, length = batch.text
            labels = batch.label
            mask = length_to_mask(length)
            loading_time_meter.update(time.time() - start)

            pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = \
                model(tokens, mask, labels)
            entropy = entropy.mean()
            normalized_entropy = normalized_entropy.mean()

            accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
            n = mask.shape[0]
            accuracy_meter.update(accuracy.item(), n)
            ce_loss_meter.update(ce_loss.item(), n)
            entropy_meter.update(entropy.item(), n)
            n_entropy_meter.update(normalized_entropy.item(), n)
            batch_time_meter.update(time.time() - start)
            start = time.time()

    logger.info(f"Valid: epoch: {epoch} ce_loss: {ce_loss_meter.avg:.4f} accuracy: {accuracy_meter.avg:.4f} "
                f"entropy: {entropy_meter.avg:.4f} n_entropy: {n_entropy_meter.avg:.4f} "
                f"loading_time: {loading_time_meter.avg:.4f} batch_time: {batch_time_meter.avg:.4f}")

    summary_writer["valid"].add_scalar(tag="ce", scalar_value=ce_loss_meter.avg, global_step=global_step)
    summary_writer["valid"].add_scalar(tag="accuracy", scalar_value=accuracy_meter.avg, global_step=global_step)
    summary_writer["valid"].add_scalar(tag="n_entropy", scalar_value=n_entropy_meter.avg, global_step=global_step)

    model.train()
    return accuracy_meter.avg


def train(train_data, valid_data, model, optimizer, lr_scheduler, es, epoch, args, logger, summary_writer):
    loading_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    entropy_meter = AverageMeter()
    n_entropy_meter = AverageMeter()
    prob_ratio_meter = AverageMeter()

    device = args.gpu_id
    model.train()
    start = time.time()
    for batch_idx, batch in enumerate(train_data):
        tokens, length = batch.text
        labels = batch.label
        mask = length_to_mask(length)
        loading_time_meter.update(time.time() - start)

        pred_labels, ce_loss, rewards, actions, actions_log_prob, entropy, normalized_entropy = \
            model(tokens, mask, labels)

        ce_loss.backward()
        perform_env_optimizer_step(optimizer, model, args)
        for k in range(args.ppo_updates):
            if k == 0:
                new_normalized_entropy, new_actions_log_prob = normalized_entropy, actions_log_prob
            else:
                new_normalized_entropy, new_actions_log_prob = model.evaluate_actions(tokens, mask, actions)
            prob_ratio = (new_actions_log_prob - actions_log_prob.detach()).exp()
            clamped_prob_ratio = prob_ratio.clamp(1.0 - args.epsilon, 1.0 + args.epsilon)
            ppo_loss = torch.max(prob_ratio * rewards, clamped_prob_ratio * rewards).mean()
            loss = ppo_loss - args.entropy_weight * new_normalized_entropy.mean()
            loss.backward()
            perform_policy_optimizer_step(optimizer, model, args)

        entropy = entropy.mean()
        normalized_entropy = normalized_entropy.mean()
        n = mask.shape[0]
        accuracy = (labels == pred_labels).to(dtype=torch.float32).mean()
        accuracy_meter.update(accuracy.item(), n)
        ce_loss_meter.update(ce_loss.item(), n)
        entropy_meter.update(entropy.item(), n)
        n_entropy_meter.update(normalized_entropy.item(), n)
        prob_ratio_meter.update((1.0-prob_ratio.detach()).abs().mean().item(), n)
        batch_time_meter.update(time.time() - start)

        global global_step
        summary_writer["train"].add_scalar(tag="ce", scalar_value=ce_loss.item(), global_step=global_step)
        summary_writer["train"].add_scalar(tag="accuracy", scalar_value=accuracy.item(), global_step=global_step)
        summary_writer["train"].add_scalar(tag="n_entropy", scalar_value=normalized_entropy.item(),
                                           global_step=global_step)
        summary_writer["train"].add_scalar(tag="prob_ratio", scalar_value=prob_ratio_meter.value,
                                           global_step=global_step)
        global_step += 1

        if (batch_idx + 1) % (len(train_data) // 3) == 0:
            logger.info(f"Train: epoch: {epoch} batch_idx: {batch_idx + 1} ce_loss: {ce_loss_meter.avg:.4f} "
                        f"accuracy: {accuracy_meter.avg:.4f} entropy: {entropy_meter.avg:.4f} "
                        f"n_entropy: {n_entropy_meter.avg:.4f} loading_time: {loading_time_meter.avg:.4f} "
                        f"batch_time: {batch_time_meter.avg:.4f}")
            val_accuracy = validate(valid_data, model, epoch, device, logger, summary_writer)
            lr_scheduler["env"].step(val_accuracy)
            lr_scheduler["policy"].step(val_accuracy)
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
    train_data, valid_data, test_data, vectors = get_data(args, logger)

    model = PpoModel(vocab_size=args.vocab_size,
                     word_dim=args.word_dim,
                     hidden_dim=args.hidden_dim,
                     mlp_hidden_dim=args.mlp_hidden_dim,
                     label_dim=args.label_size,
                     dropout_prob=args.dropout_prob,
                     parser_leaf_transformation=args.parser_leaf_transformation,
                     parser_trans_hidden_dim=args.parser_trans_hidden_dim,
                     tree_leaf_transformation=args.tree_leaf_transformation,
                     tree_trans_hidden_dim=args.tree_trans_hidden_dim,
                     baseline_type=args.baseline_type,
                     var_normalization=args.var_normalization,
                     use_batchnorm=args.use_batchnorm).cuda(args.gpu_id)

    if vectors is not None:
        model.embd_parser.weight.data = vectors.cuda(args.gpu_id).clone()
        model.embd_tree.weight.data = vectors.cuda(args.gpu_id).clone()
    if args.freeze_embeddings:
        model.embd_parser.weight.requires_grad = False
        model.embd_tree.weight.requires_grad = False
        logger.info("Embeddings is frozen!")

    optimizer, lr_scheduler, es = prepare_optimisers(args, logger,
                                                     policy_parameters=model.get_policy_parameters(),
                                                     environment_parameters=model.get_environment_parameters())

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
    # # SST-2
    args = {"vectors":                    "glove.840B.300d",
            "freeze-embeddings":          "False",
            "fine-grained":               "False",
            "use-batchnorm":              "True",
            "dropout-prob":               0.5,
            "lower":                      "True",
            "mlp-hidden-dim":             300,
            "word-dim":                   300,
            "hidden-dim":                 300,
            "parser-leaf-transformation": "no_transformation",
            "parser-trans-hidden_dim":    300,
            "tree-leaf-transformation":   "lstm_transformation",
            "tree-trans-hidden_dim":      300,
            "baseline-type":              "self_critical",
            "var-normalization":          "True",
            "entropy-weight":             0.0,
            "clip-grad-norm":             0.0,
            "env-optimizer":              "adadelta",
            "pol-optimizer":              "adadelta",
            "env-lr":                     1.0,
            "pol-lr":                     1.0,
            "ppo-updates":                5,
            "epsilon":                    0.2,
            "lr-scheduler-patience":      8,
            "lr-scheduler-threshold":     0.005,
            "l2-weight":                  0.0,
            "batch-size":                 64,
            "max-epoch":                  150,
            "es-patience":                20,
            "es-threshold":               0.005,
            "gpu-id":                     0,
            "model-dir":                  "data/sst2/ppo/models/exp0",
            "logs-path":                  "data/sst2/ppo/logs/exp0",
            "tensorboard-path":           "data/sst2/ppo/tensorboard/exp0"
    }

    # SST-5
    # args = {"vectors":                    "glove.840B.300d",
    #         "freeze-embeddings":          "False",
    #         "fine-grained":               "True",
    #         "use-batchnorm":              "True",
    #         "dropout-prob":               0.5,
    #         "lower":                      "False",
    #         "mlp-hidden-dim":             300,
    #         "word-dim":                   300,
    #         "hidden-dim":                 300,
    #         "parser-leaf-transformation": "no_transformation",
    #         "parser-trans-hidden_dim":    300,
    #         "tree-leaf-transformation":   "lstm_transformation",
    #         "tree-trans-hidden_dim":      300,
    #         "baseline-type":              "self_critical",
    #         "var-normalization":          "True",
    #         "entropy-weight":             0.0,
    #         "clip-grad-norm":             0.0,
    #         "env-optimizer":              "adadelta",
    #         "pol-optimizer":              "adadelta",
    #         "env-lr":                     1.0,
    #         "pol-lr":                     1.0,
    #         "ppo-updates":                1,
    #         "epsilon":                    0.2,
    #         "lr-scheduler-patience":      8,
    #         "lr-scheduler-threshold":     0.005,
    #         "l2-weight":                  0.0,
    #         "batch-size":                 32,
    #         "max-epoch":                  150,
    #         "es-patience":                20,
    #         "es-threshold":               0.005,
    #         "gpu-id":                     0,
    #         "model-dir":                  "data/sst5/ppo/models/exp0",
    #         "logs-path":                  "data/sst5/ppo/logs/exp0",
    #         "tensorboard-path":           "data/sst5/ppo/tensorboard/exp0"
    # }

    parser = argparse.ArgumentParser()

    parser.add_argument("--vectors", default=args["vectors"], type=lambda val: None if val == "None" else val)
    parser.add_argument("--freeze-embeddings", default=args["freeze-embeddings"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--fine-grained", default=args["fine-grained"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--use-batchnorm", default=args["use-batchnorm"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--dropout-prob", default=args["dropout-prob"], type=float)
    parser.add_argument("--lower", default=args["lower"],
                        type=lambda val: True if val == "True" else False)
    parser.add_argument("--mlp-hidden-dim", default=args["mlp-hidden-dim"], type=int)


    parser.add_argument("--word-dim", required=False, default=args["word-dim"], type=int)
    parser.add_argument("--hidden-dim", required=False, default=args["hidden-dim"], type=int)
    parser.add_argument("--parser-leaf-transformation", required=False, default=args["parser-leaf-transformation"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--parser-trans-hidden_dim", required=False, default=args["parser-trans-hidden_dim"], type=int)
    parser.add_argument("--tree-leaf-transformation", required=False, default=args["tree-leaf-transformation"],
                        choices=["no_transformation", "lstm_transformation",
                                 "bi_lstm_transformation", "conv_transformation"])
    parser.add_argument("--tree-trans-hidden_dim", required=False, default=args["tree-trans-hidden_dim"], type=int)

    parser.add_argument("--baseline-type", default=args["baseline-type"],
                        choices=["no_baseline", "ema", "self_critical"])
    parser.add_argument("--var-normalization", default=args["var-normalization"],
                        type=lambda string: True if string == "True" else False)
    parser.add_argument("--entropy-weight", default=args["entropy-weight"], type=float)
    parser.add_argument("--clip-grad-norm", default=args["clip-grad-norm"], type=float,
                        help="If the value is less or equal to zero clipping is not performed.")

    parser.add_argument("--env-optimizer", required=False, default=args["env-optimizer"], choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--pol-optimizer", required=False, default=args["pol-optimizer"], choices=["adam", "amsgrad", "sgd", "adadelta"])
    parser.add_argument("--env-lr", required=False, default=args["env-lr"], type=float)
    parser.add_argument("--pol-lr", required=False, default=args["pol-lr"], type=float)
    parser.add_argument("--ppo-updates", required=False, default=args["ppo-updates"], type=int)
    parser.add_argument("--epsilon", required=False, default=args["epsilon"], type=float)
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
