import collections
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import layers


class MultiHeadAttention(nn.Layer):
    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadAttention, self).__init__()
        # 输入的embedding维度
        self.embed_dim = embed_dim
        # key的维度
        self.kdim = kdim if kdim is not None else embed_dim
        # value的维度
        self.vdim = vdim if vdim is not None else embed_dim
        # head的数目
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # query
        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        # key
        self.k_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        # value
        self.v_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v

        q, k, v = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim ** -0.5)
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)

        return out if len(outs) == 1 else tuple(outs)


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


def _convert_param_attr_to_list(param_attr, n):
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
                "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
        # param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        # multi head attention
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])

        # feed forward
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src

        #  multi head attention
        src = self.self_attn(src, src, src, src_mask)
        # 残差连接
        src = residual + self.dropout1(src)
        # Norm
        src = self.norm1(src)
        residual = src

        # feed forward
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 残差连接
        src = residual + self.dropout2(src)
        # Norm
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)
        # multi head attention
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        # encoder decoder attention
        self.cross_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])
        # feed forward
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        # 残差连接
        tgt = residual + self.dropout1(tgt)
        # Norm
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory,
                                                memory_mask, cache[1])
        # 残差连接
        tgt = residual + self.dropout2(tgt)
        # Norm
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 残差连接
        tgt = residual + self.dropout3(tgt)
        # Norm
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))



# 安装依赖
# !pip install -r requirements.txt
get_ipython().system('pip install subword_nmt==0.3.7')
get_ipython().system('pip install attrdict==2.0.1')
get_ipython().system('pip install paddlenlp==2.0.0rc22')


import os
import time
import yaml
import logging
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict
import jieba
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from functools import partial

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader
from paddlenlp.data import Vocab, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, \
    position_encoding_init
from paddlenlp.utils.log import logger



zh_dir = 'en-zh/train.zh'
en_dir = 'en-zh/train.en'


# 中文需要Jieba+BPE，英文需要BPE
# 中文Jieba分词
def jieba_cut(in_file, out_file):
    out_f = open(out_file, 'w', encoding='utf8')
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            cut_line = ' '.join(jieba.cut(line))
            out_f.write(cut_line + '\n')
    out_f.close()


cut_zh_dir = 'en-zh/train.zh.cut.txt'
jieba_cut(zh_dir, cut_zh_dir)


zh_dir = 'dev_cn.txt'
cut_zh_dir = 'dev_cn.cut.txt'
jieba_cut(zh_dir, cut_zh_dir)



print('generate the training data')
get_ipython().system('subword-nmt learn-bpe -s 32000 < en-zh/train.zh.cut.txt > en-zh/bpe.ch.32000')
get_ipython().system('subword-nmt learn-bpe -s 32000 < en-zh/train.en > en-zh/bpe.en.32000')



get_ipython().system('subword-nmt apply-bpe -c en-zh/bpe.ch.32000 < en-zh/train.zh.cut.txt > en-zh/train.ch.bpe')
get_ipython().system('subword-nmt apply-bpe -c en-zh/bpe.ch.32000 < dev_cn.cut.txt > en-zh/dev.ch.bpe')
get_ipython().system('subword-nmt apply-bpe -c en-zh/bpe.en.32000 < en-zh/train.en > en-zh/train.en.bpe')
get_ipython().system('subword-nmt apply-bpe -c en-zh/bpe.en.32000 < dev_en.txt > en-zh/dev.en.bpe')



get_ipython().system('subword-nmt  get-vocab -i en-zh/train.ch.bpe -o en-zh/temp')



special_token = ['<s>', '<e>', '<unk>']
cn_vocab = []
with open('en-zh/temp') as f:
    for item in f.readlines():
        words = item.strip().split()
        cn_vocab.append(words[0])

with open('en-zh/vocab.ch.tgt', 'w') as f:
    for item in special_token:
        f.write(item + '\n')
    for item in cn_vocab:
        f.write(item + '\n')


get_ipython().system('subword-nmt  get-vocab -i en-zh/train.en.bpe -o en-zh/temp')



eng_vocab = []
with open('en-zh/temp') as f:
    for item in f.readlines():
        words = item.strip().split()
        eng_vocab.append(words[0])

with open('en-zh/vocab.en.src', 'w') as f:
    for item in special_token:
        f.write(item + '\n')
    for item in eng_vocab:
        f.write(item + '\n')


cn_data = []
with open('en-zh/train.ch.bpe') as f:
    for item in f.readlines():
        words = item.strip()
        cn_data.append(words)
en_data = []
with open('en-zh/train.en.bpe') as f:
    for item in f.readlines():
        words = item.strip()
        en_data.append(words)

print(cn_data[:10])
print(en_data[:10])


def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def read(src_path, tgt_path, is_predict=False):
    if is_predict:
        with open(src_path, 'r', encoding='utf8') as src_f:
            for src_line in src_f.readlines():
                src_line = src_line.strip()
                if not src_line:
                    continue
                yield {'src': src_line, 'tgt': ''}
    else:
        with open(src_path, 'r', encoding='utf8') as src_f, open(tgt_path, 'r', encoding='utf8') as tgt_f:
            for src_line, tgt_line in zip(src_f.readlines(), tgt_f.readlines()):
                src_line = src_line.strip()
                if not src_line:
                    continue
                tgt_line = tgt_line.strip()
                if not tgt_line:
                    continue
                yield {'src': src_line, 'tgt': tgt_line}


# 创建训练集、验证集的dataloader
def create_data_loader(args):
    train_dataset = load_dataset(read, src_path=args.training_file.split(',')[0],
                                 tgt_path=args.training_file.split(',')[1], lazy=False)
    dev_dataset = load_dataset(read, src_path=args.training_file.split(',')[0],
                               tgt_path=args.training_file.split(',')[1], lazy=False)
    print('load src vocab')
    print(args.src_vocab_fpath)
    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    print('load trg vocab')
    print(args.trg_vocab_fpath)
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    print('padding')
    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))
    print('convert example')

    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    data_loaders = [(None)] * 2
    print('dataset loop')
    for i, dataset in enumerate([train_dataset, dev_dataset]):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(
                min_max_filer, max_len=args.max_length))

        sampler = SamplerHelper(dataset)

        if args.sort_type == SortType.GLOBAL:
            src_key = (lambda x, data_source: len(data_source[x][0]) + 1)
            trg_key = (lambda x, data_source: len(data_source[x][1]) + 1)
            # Sort twice
            sampler = sampler.sort(key=trg_key).sort(key=src_key)
        else:
            if args.shuffle:
                sampler = sampler.shuffle(seed=args.shuffle_seed)
            max_key = (lambda x, data_source: max(len(data_source[x][0]), len(data_source[x][1])) + 1)
            if args.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=args.pool_size)

        batch_size_fn = lambda new, count, sofar, data_source: max(sofar, len(data_source[new][0]) + 1,
                                                                   len(data_source[new][1]) + 1)
        batch_sampler = sampler.batch(
            batch_size=args.batch_size,
            drop_last=False,
            batch_size_fn=batch_size_fn,
            key=lambda size_so_far, minibatch_len: size_so_far * minibatch_len)

        if args.shuffle_batch:
            batch_sampler = batch_sampler.shuffle(seed=args.shuffle_seed)

        if i == 0:
            batch_sampler = batch_sampler.shard()

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                pad_idx=args.bos_idx),
            num_workers=2,
            return_list=True)
        data_loaders[i] = (data_loader)
    return data_loaders


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


# !pwd

yaml_file = './transformer.base.yaml'
with open(yaml_file, 'rt') as f:
    args = AttrDict(yaml.safe_load(f))
    pprint(args)


print(args.training_file.split(','))


def prepare_train_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])
    trg_word = word_pad([[bos_idx] + inst[1] for inst in insts])
    lbl_word = np.expand_dims(
        word_pad([inst[1] + [eos_idx] for inst in insts]), axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs



# Define data loader
(train_loader), (eval_loader) = create_data_loader(args)


for input_data in train_loader:
    (src_word, trg_word, lbl_word) = input_data
    print(src_word)
    print(trg_word)
    print(lbl_word)
    break



def do_train(args, train_loader, eval_loader):
    if args.use_gpu:
        rank = dist.get_rank()
        trainer_count = dist.get_world_size()
    else:
        rank = 0
        trainer_count = 1
        paddle.set_device("cpu")

    if trainer_count > 1:
        dist.init_parallel_env()

    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define model
    transformer = TransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    scheduler = paddle.optimizer.lr.NoamDecay(
        args.d_model, args.warmup_steps, args.learning_rate, last_epoch=0)

    # Define optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=float(args.eps),
        parameters=transformer.parameters())

    # Init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        model_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "transformer.pdparams"))
        opt_dict = paddle.load(
            os.path.join(args.init_from_checkpoint, "transformer.pdopt"))
        transformer.set_state_dict(model_dict)
        optimizer.set_state_dict(opt_dict)
        print("loaded from checkpoint.")
    # Init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        model_dict = paddle.load(
            os.path.join(args.init_from_pretrain_model, "transformer.pdparams"))
        transformer.set_state_dict(model_dict)
        print("loaded from pre-trained model.")

    if trainer_count > 1:
        transformer = paddle.DataParallel(transformer)

    # The best cross-entropy value with label smoothing
    loss_normalizer = -(
            (1. - args.label_smooth_eps) * np.log(
        (1. - args.label_smooth_eps)) + args.label_smooth_eps *
            np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))

    ce_time = []
    ce_ppl = []
    step_idx = 0

    # Train loop
    for pass_id in range(args.epoch):
        epoch_start = time.time()

        batch_id = 0
        batch_start = time.time()
        for input_data in train_loader:
            (src_word, trg_word, lbl_word) = input_data

            logits = transformer(src_word=src_word, trg_word=trg_word)

            sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

            avg_cost.backward()

            optimizer.step()
            optimizer.clear_grad()

            if step_idx % args.print_step == 0 and rank == 0:
                total_avg_cost = avg_cost.numpy()

                if step_idx == 0:
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f " %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)])))
                else:
                    train_avg_batch_cost = args.print_step / (
                            time.time() - batch_start)
                    logger.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, avg_speed: %.2f step/sec"
                        % (
                            step_idx,
                            pass_id,
                            batch_id,
                            total_avg_cost,
                            total_avg_cost - loss_normalizer,
                            np.exp([min(total_avg_cost, 100)]),
                            train_avg_batch_cost,))
                batch_start = time.time()

            if step_idx % args.save_step == 0 and step_idx != 0:
                # Validation
                transformer.eval()
                total_sum_cost = 0
                total_token_num = 0
                with paddle.no_grad():
                    for input_data in eval_loader:
                        (src_word, trg_word, lbl_word) = input_data
                        logits = transformer(
                            src_word=src_word, trg_word=trg_word)
                        sum_cost, avg_cost, token_num = criterion(logits,
                                                                  lbl_word)
                        total_sum_cost += sum_cost.numpy()
                        total_token_num += token_num.numpy()
                        total_avg_cost = total_sum_cost / total_token_num
                    logger.info("validation, step_idx: %d, avg loss: %f, "
                                "normalized loss: %f, ppl: %f" %
                                (step_idx, total_avg_cost,
                                 total_avg_cost - loss_normalizer,
                                 np.exp([min(total_avg_cost, 100)])))
                transformer.train()

                if args.save_model and rank == 0:
                    model_dir = os.path.join(args.save_model,
                                             "step_" + str(step_idx))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(transformer.state_dict(),
                                os.path.join(model_dir, "transformer.pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(model_dir, "transformer.pdopt"))
                batch_start = time.time()
            batch_id += 1
            step_idx += 1
            scheduler.step()

        train_epoch_cost = time.time() - epoch_start
        ce_time.append(train_epoch_cost)
        logger.info("train epoch: %d, epoch_cost: %.5f s" %
                    (pass_id, train_epoch_cost))

    if args.save_model and rank == 0:
        model_dir = os.path.join(args.save_model, "step_final")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.save(transformer.state_dict(),
                    os.path.join(model_dir, "transformer.pdparams"))
        paddle.save(optimizer.state_dict(),
                    os.path.join(model_dir, "transformer.pdopt"))


print('training the model')
do_train(args, train_loader, eval_loader)
