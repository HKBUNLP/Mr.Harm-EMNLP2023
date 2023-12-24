import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from .dist_utils import all_gather

SMALL_NUM = np.log(1e-45)

def compute_clm(pl_module, ret):
    clm_loss = ret["text_outputs"].loss

    new_ret = {
        f"clm_loss": clm_loss
    }

    phase = "train" if pl_module.training else "val"
    loss_clm = getattr(pl_module, f"{phase}_clm_loss")(clm_loss)
    pl_module.log(f"clm/{phase}/clm_loss", loss_clm)
    return new_ret

def compute_mim(pl_module, ret, mode):
    reconstructed_pixel_values = ret[f"{mode}_logits"]
    image_features = ret["image_features"]
    if "self" in mode:
        image_masks = ret["encoder_image_masks"]
    else:
        image_masks = ret["decoder_image_masks"]

    size = pl_module.hparams.config["image_size"] // pl_module.hparams.config["patch_size"]
    bool_masked_pos = image_masks.reshape(-1, size, size)
    mask = (
        bool_masked_pos.repeat_interleave(pl_module.hparams.config["patch_size"], 1)
        .repeat_interleave(pl_module.hparams.config["patch_size"], 2)
        .unsqueeze(1)
        .contiguous()
    )
    reconstruction_loss = nn.functional.l1_loss(
        image_features, reconstructed_pixel_values, reduction="none"
    )
    mim_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / 3
    
    new_ret = {
        f"{mode}_mim_loss": mim_loss
    }

    phase = "train" if pl_module.training else "val"
    loss_mim = getattr(pl_module, f"{phase}_{mode}_loss")(mim_loss)
    pl_module.log(f"{mode}/{phase}/{mode}_loss", loss_mim)
    return new_ret


def compute_contrastive(pl_module, ret):
    # Query
    text_reps = F.normalize(ret["text_bottleneck_repre"])
    image_reps = F.normalize(ret["image_bottleneck_repre"])
    
    all_text_reps = pl_module.gather(text_reps)
    all_image_reps = pl_module.gather(image_reps)

    # in-batch contrastive
    # Cross Entropy
    logits_per_text = torch.einsum("nc,ck->nk", [all_text_reps, all_image_reps.transpose(-2, -1)]) / pl_module.T
    contrastive_loss = clip_loss(logits_per_text)

    new_ret = {
        "contrastive_loss": contrastive_loss,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_contrastive_loss")(new_ret["contrastive_loss"])
    pl_module.log(f"contrastive/{phase}/loss", loss)

    return new_ret

@torch.no_grad()
def compute_irtr_recall(pl_module):
    ###
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)
    ###

    text_dset = pl_module.trainer.datamodule.dms[0].make_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        # # print(_b)
        # texts = clip.tokenize(_b["text"], truncate=True).to(device)
        # text_features = model.encode_text(texts)
        # # assert 1 == 2
        text_ids = _b["text_ids"].to(pl_module.device)
        text_masks = _b["text_masks"].to(pl_module.device)
        text_preload.append(
            {
                "img_index": _b["img_index"],
                "text_reps": pl_module.encode_text(
                text_ids, text_masks)[0]
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)
    
    image_preload = dict()
    image_preload_reps = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        img_index = _b["img_index"][0]
        if img_index not in image_preload:
            # ###
            # img_features = []
            # for img_dir in _b['img_dirs']:
            #     img_feature = preprocess(Image.open(img_dir)).unsqueeze(0).to(device)
            #     img_features.append(img_feature)
            # img_features = torch.cat(img_features, dim=0)
            # ###
            # img_reps = model.encode_image(img_features)

            image_features = _b["image_features"].to(pl_module.device)
            img_reps = pl_module.encode_image(image_features)[0] # [bsz, 768]
            image_preload[img_index] = 1
            image_preload_reps.append((img_reps, _b["img_index"]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload_reps, desc="rank loop"):
        _img_reps, _iid = img_batch # [bsz, 768]
        _img_reps = _img_reps / torch.norm(_img_reps, dim=-1, keepdim=True)

        img_batch_score = list()
        for txt_batch in text_preload:
            _text_reps = txt_batch["text_reps"] # [bsz, 768]
            _text_reps = _text_reps / torch.norm(_text_reps, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast():
                score = torch.einsum('nc,cm->nm', [_img_reps, _text_reps.transpose(-1, -2)])
            img_batch_score.append(score)
        img_batch_score = torch.cat(img_batch_score, dim=-1) # [bsz, num_texts]
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids += _iid
    
    ###
    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    # scores = torch.cat(rank_scores, dim=0) # [5000, 25010]
    # iids = torch.tensor(rank_iids).view(-1) # all image ids, [5000]
    ###

    topk5 = scores.topk(5, dim=0)
    topk5_iids = iids[topk5.indices] # [5, 25010]
    # print(topk5.values[:, 20:25])
    # print(topk5_iids[:, 20:25])
    # assert 1 == 2

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices] # [5000, 10]
    topk5_iids = tiids[topk5.indices] # [5000, 5]
    topk1_iids = tiids[topk1.indices] # [5000, 1]


    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices] # [10, 25010]
    topk5_iids = iids[topk5.indices] # [5, 25010]
    topk1_iids = iids[topk1.indices] # [1, 25010]
    # tiids [25010]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
    # print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10))

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()