from email.errors import NonPrintableDefect
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import json
import jsonlines

from torch import distributed as dist
from transformers import CLIPVisionModel, T5Tokenizer

from . import mm_utils
from . import heads, objectives
from . import dist_utils
from .t5_model import T5ForMultimodalGeneration

torch.backends.cudnn.enabled = False

class MMTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.mode = self.hparams.config["mode"]
        self.out_path = self.hparams.config["out_path"]

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                CLIPVisionModel.from_pretrained(config["vit"])
                T5ForMultimodalGeneration.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        
        #####################################################################################
        self.image_transformer = CLIPVisionModel.from_pretrained(config["vit"])
        self.text_transformer = T5ForMultimodalGeneration.from_pretrained(
            config['tokenizer'],
            config["input_image_embed_size"],
        )
        self.tokenizer = T5Tokenizer.from_pretrained(config['tokenizer'])
        #####################################################################################
        for param in self.image_transformer.parameters():
            param.requires_grad = False

        mm_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load model ======================
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        
        self.pred_result = {}
        self.gold_result = {}
        
    def encode_image(
        self,
        image_features,
    ):
        last_hidden_state = self.image_transformer(
            pixel_values=image_features,
        ).last_hidden_state
        return last_hidden_state
    
    def infer(
        self,
        batch,
    ):
        text_ids = batch[f"text_ids"]
        label_ids = batch[f"label_ids"] if self.mode != "rationale" or "rationale_ids" not in batch else batch[f"rationale_ids"]
        label_ids[label_ids==0] = -100
        text_masks = batch[f"text_masks"]
        image_features = batch[f"image_features"]

        image_features = self.encode_image(image_features)
        text_outputs = self.text_transformer(
            input_ids=text_ids,
            attention_mask=text_masks,
            image_ids=image_features,
            labels=label_ids,
        )

        ret = {
            "text_outputs": text_outputs,
        }

        return ret

    def forward(self, batch):
        ret = dict()

        ret.update(self.infer(batch))

        ret.update(objectives.compute_clm(self, ret))
        
        return ret

    def training_step(self, batch, batch_idx):
        mm_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        mm_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        mm_utils.set_task(self)
        if self.mode != "rationale":
            text_ids = batch[f"text_ids"]
            image_features = batch[f"image_features"]
            image_features = self.encode_image(image_features)
            self.text_transformer.encoder.update_image_ids(image_features)
            self.text_transformer.update_image_ids(image_features)
            outputs = self.text_transformer.generate(text_ids, max_length=256)
            pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for iid in range(len(pred)):
                self.pred_result[batch["img_index"][iid]] = pred[iid]
                self.gold_result[batch["img_index"][iid]] = batch["label"][iid].split("The answer is: ")[-1].strip()
            ret = dict()
        else:
            ret = self(batch)

        return ret

    def validation_epoch_end(self, outs):
        if self.mode != "rationale":
            correct = 0
            for iid in self.gold_result:
                if iid not in self.pred_result:
                    correct = 0
                    break
                label = self.gold_result[iid]
                pred = self.pred_result[iid].split("The answer is: ")[-1].strip()
                if pred == label:
                    correct += 1
            self.acc = correct / len(self.gold_result)
            self.pred_result = {}
        mm_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        mm_utils.set_task(self)

        text_ids = batch[f"text_ids"]
        image_features = batch[f"image_features"]
        image_features = self.encode_image(image_features)
        self.text_transformer.encoder.update_image_ids(image_features)
        self.text_transformer.update_image_ids(image_features)
        outputs = self.text_transformer.generate(text_ids, max_length=256)
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        ret = dict()
        self.pred_result[batch["img_index"][0]] = pred

        return ret

    def test_epoch_end(self, outs):
        with open(self.out_path, "w") as fout:
            json.dump(self.pred_result, fout)
        mm_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return mm_utils.set_schedule(self)