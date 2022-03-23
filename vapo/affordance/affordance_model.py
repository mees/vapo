import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from vapo.affordance.hough_voting import hough_voting as hv
from vapo.affordance.utils.losses import (
    compute_dice_loss,
    compute_dice_score,
    compute_mIoU,
    CosineSimilarityLossWithMask,
    get_affordance_loss,
)


class AffordanceModel(pl.LightningModule):
    def __init__(self, cfg, input_channels=1, n_classes=2, cmd_log=None, *args, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        # https://github.com/qubvel/segmentation_models.pytorch
        self.unet, self.center_direction_net = self.init_model(
            decoder_channels=cfg.unet_cfg.decoder_channels,
            in_channels=input_channels,
            n_classes=self.n_classes,
        )
        self.optimizer_cfg = cfg.optimizer
        # Loss function
        self.affordance_loss = get_affordance_loss(cfg.loss, self.n_classes)
        self.center_loss = CosineSimilarityLossWithMask(weighted=True)
        self.loss_w = cfg.loss

        # Misc
        self.cmd_log = cmd_log
        self._batch_loss = []
        self._batch_miou = []

        # Hough Voting stuff (this operates on CUDA only)
        self.hough_voting_layer = hv.HoughVoting(**cfg.hough_voting)
        print("hvl init")

        # Prediction act_fnc
        if self.n_classes > 1:
            # Softmax over channels
            self.act_fnc = torch.nn.Softmax(1)
        else:
            self.act_fnc = torch.nn.Sigmoid()
        self.save_hyperparameters()

    def init_model(self, decoder_channels=None, n_classes=2, in_channels=1):
        if decoder_channels is None:
            decoder_channels = [128, 64, 32]
        # encoder_depth Should be equal to number of layers in decoder
        unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,  # Grayscale
            classes=n_classes,
            encoder_depth=len(decoder_channels),
            decoder_channels=tuple(decoder_channels),
            activation=None,
        )
        # Fix encoder weights. Only train decoder
        for param in unet.encoder.parameters():
            param.requires_grad = False

        # A 1x1 conv layer that goes from embedded features to 2d pixel direction
        feature_dim = decoder_channels[-1]
        center_direction_net = nn.Conv2d(feature_dim, 2, kernel_size=1, stride=1, padding=0, bias=False)

        return unet, center_direction_net

    def _calc_aff_loss(self, preds, labels):
        # Preds = (B, C, H, W)
        # labels = (B, H, W)
        B, C, H, W = preds.shape
        if C == 1:
            # BCE needs B, H, W
            preds = preds.squeeze(1)
            labels = labels.float()
        ce_loss = self.affordance_loss(preds, labels)
        info = {"CE_loss": ce_loss}
        loss = self.loss_w.ce_loss * ce_loss

        # Add dice if required
        if self.loss_w.affordance.add_dice:
            if C == 1:
                # Dice needs B, C, H, W
                preds = preds.unsqueeze(1)
                labels = labels.unsqueeze(1)
            # label_spatial = pixel2spatial(labels.long(), H, W)
            dice_loss = compute_dice_loss(labels.long(), preds)
            info["dice_loss"] = dice_loss
            loss += self.loss_w.dice * dice_loss
        return loss, info

    def compute_loss(self, preds, labels):
        # Activation fnc is applied in loss fnc hence, use logits
        # Affordance loss
        aff_loss, info = self._calc_aff_loss(preds["affordance_logits"], labels["affordance"])

        # Center prediction loss
        if self.n_classes > 2:
            bin_mask = torch.zeros_like(labels["affordance"])
            bin_mask[labels["affordance"] > 0] = 1
        else:
            bin_mask = labels["affordance"]
        center_loss = self.center_loss(preds["center_dirs"], labels["center_dirs"], bin_mask)

        info.update({"center_loss": center_loss})

        # Total loss
        total_loss = aff_loss + self.loss_w.centers * center_loss
        return total_loss, info

    def eval_mode(self):
        self.unet.eval()
        self.center_direction_net.eval()

    def train_mode(self):
        self.unet.train()
        self.center_direction_net.train()

    # Affordance mask prediction
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        aff_logits = self.unet.segmentation_head(decoder_output)
        center_direction_prediction = self.center_direction_net(decoder_output)

        # Affordance
        aff_probs = self.act_fnc(aff_logits)  # [N x C x H x W]
        aff_mask = torch.argmax(aff_probs, dim=1)  # [N x H x W]
        return aff_logits, aff_probs, aff_mask, center_direction_prediction

    # Center prediction
    def get_centers(self, aff_mask, directions):
        """
        :param aff_mask (torch.tensor, int64): [N x H x W]
        :param directions (torch.tensor, float32): [N x 2 x H x W]

        :return object_centers (list(torch.tensor), int64)
        :return directions (torch.tensor, float32): [N x 2 x H x W] normalized directions
        :return initial_masks (torch.tensor): [N x 1 x H x W]
            - range: int values indicating object mask (0 to n_objects)
        """
        # x.shape = (B, C, H, W)
        if self.n_classes > 2:
            if len(aff_mask.shape) == 4:
                bin_mask = aff_mask.any(1)
            else:
                bin_mask = torch.zeros_like(aff_mask)
                bin_mask[aff_mask > 0] = 1
        else:
            bin_mask = aff_mask

        with torch.no_grad():
            # Center direction
            directions /= torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-10)
            directions = directions.float()
            initial_masks, num_objects, object_centers_padded = self.hough_voting_layer(
                (bin_mask == 1).int(), directions
            )

        # Compute list of object centers
        object_centers = []
        for i in range(initial_masks.shape[0]):
            centers_padded = object_centers_padded[i]
            centers_padded = centers_padded.permute((1, 0))[: num_objects[i], :]
            for obj_center in centers_padded:
                if torch.norm(obj_center) > 0:
                    # cast to int for pixel
                    object_centers.append(obj_center.long())
        return object_centers, directions, initial_masks

    def log_stats(self, split, max_batch, batch_idx, loss, miou):
        if batch_idx >= max_batch - 1:
            e_loss = 0 if len(self._batch_loss) == 0 else np.mean(self._batch_loss)
            e_miou = 0 if len(self._batch_miou) == 0 else np.mean(self._batch_miou)
            self.cmd_log.info(
                "%s [epoch %4d]" % (split, self.current_epoch) + "loss: %.3f, mIou: %.3f" % (e_loss, e_miou)
            )
            self._batch_loss = []
            self._batch_miou = []
        else:
            self._batch_loss.append(loss.item())
            self._batch_miou.append(miou.item())

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, labels = batch

        # B, N_classes, img_size, img_size
        # Forward pass
        aff_logits, _, _, center_dir = self.forward(x)
        preds = {"affordance_logits": aff_logits, "center_dirs": center_dir}

        # Compute loss
        total_loss, info = self.compute_loss(preds, labels)

        # Metrics
        mIoU = compute_mIoU(aff_logits, labels["affordance"])
        dice_score = compute_dice_score(aff_logits, labels["affordance"])

        self.log_stats("train", self.trainer.num_training_batches, batch_idx, total_loss, mIoU)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_dice_score", dice_score, on_step=False, on_epoch=True)
        self.log("train_miou", mIoU, on_step=False, on_epoch=True)
        for k, v in info.items():
            self.log("train_%s" % k, v, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch
        # Predictions
        aff_logits, _, aff_mask, directions = self.forward(x)
        _, center_dir, _ = self.get_centers(aff_mask, directions)
        preds = {"affordance_logits": aff_logits, "center_dirs": center_dir}

        # Compute loss
        total_loss, info = self.compute_loss(preds, labels)

        # Compute metrics
        mIoU = compute_mIoU(aff_logits, labels["affordance"])
        dice_score = compute_dice_score(aff_logits, labels["affordance"])

        # Log metrics
        self.log_stats("validation", sum(self.trainer.num_val_batches), batch_idx, total_loss, mIoU)
        self.log("val_miou", mIoU, on_step=False, on_epoch=True)
        self.log("val_dice_score", dice_score, on_step=False, on_epoch=True)
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True)
        for k, v in info.items():
            self.log("val_%s" % k, v, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)
        return optimizer
