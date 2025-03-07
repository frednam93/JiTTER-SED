from tqdm import tqdm

import torch

from recipes.mlm.train import Trainer
from src.functional.loss import MSELoss
from src.preprocess.data_aug import frame_shift, feature_transformation


class MLMTrainer(Trainer):

    def __init__(self, *argc, **kwarg):
        super().__init__(*argc, **kwarg)
        self.reconstruction_loss = MSELoss()
        self.n_train = len(self.train_loader)
        self.i_sr = self.config["training"]["init_shuffle_rate"]
        self.f_sr = self.config["training"]["final_shuffle_rate"]
        self.i_epoch = self.config["training"]["epoch_shuffle_increase"]
        self.f_epoch = self.config["training"]["epoch_shuffle_saturate"]
        self.i_step = self.config["training"]["epoch_shuffle_increase"] * self.n_train
        self.f_step = self.config["training"]["epoch_shuffle_saturate"] * self.n_train

    def shuffle_rate_update(self, epoch, i):
        current_step = epoch * self.n_train + i
        if epoch < self.i_epoch:
            self.net.mlm_tool.shuffle_rate = self.i_sr
        elif epoch < self.f_epoch:
            self.net.mlm_tool.shuffle_rate = self.i_sr + (self.f_sr - self.i_sr) * (current_step - self.i_step) / (self.f_step - self.i_step)
        else:
            self.net.mlm_tool.shuffle_rate = self.f_es

    def train(self, epoch):
        self.net.train()
        tk0 = tqdm(self.train_loader, total=self.n_train, leave=False, desc="training processing")

        mean_loss = 0

        for i, (wavs, _, _, _) in enumerate(tk0, 0):
            if self.config["training"]["dynamic_shuffle"]:
                self.shuffle_rate_update(epoch, i)

            wavs = wavs.to(self.device)
            # Data preprocessing
            if self.multigpu:
                mel = self.net.module.get_feature_extractor()(wavs)
            else:
                mel = self.net.get_feature_extractor()(wavs)

            # time shift
            mel = frame_shift(mel)
            mel = feature_transformation(mel, **self.config["training"]["transform"])

            pred, other_dict = self.net(mel, encoder_win=self.config["training"]["encoder_win"])
            assert pred.shape[1] == 1000
            frame_before_mask = other_dict["frame_before_mask"]
            mask_id_seq = other_dict["mask_id_seq"]
            if self.config["training"]["train_maskid"]:
                loss = self.reconstruction_loss(frame_before_mask[mask_id_seq], pred[mask_id_seq])
            else:
                loss = self.reconstruction_loss(frame_before_mask, pred)
            if self.net.mlm_tool.multitask:
                mt_pred = other_dict["multitask_pred"]
                mt_mask = other_dict["multitask_mask"]
                if self.config["training"]["train_maskid"]:
                    loss += self.reconstruction_loss(frame_before_mask[mt_mask], mt_pred[mt_mask])
                else:
                    loss += self.reconstruction_loss(frame_before_mask, mt_pred)


            torch.nn.utils.clip_grad_norm(self.net.parameters(), max_norm=20, norm_type=2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            mean_loss += loss.item() / self.n_train

        self.logger.info("Epoch {0}: Train loss is {1}".format(epoch, mean_loss))
        self.logger.info("Epoch {0}: lr scale is {1}".format(epoch, self.scheduler._get_scale()))
        return

    def validation(self, epoch):
        self.net.eval()
        n_valid = len(self.val_loader)
        mean_loss = 0

        with torch.no_grad():
            tk1 = tqdm(self.val_loader, total=n_valid, leave=False, desc="validation processing")
            for _, (wavs, _, _, _) in enumerate(tk1, 0):
                wavs = wavs.to(self.device)
                if self.multigpu:
                    mel = self.net.module.get_feature_extractor()(wavs)
                else:
                    mel = self.net.get_feature_extractor()(wavs)
                pred, other_dict = self.net(mel, encoder_win=self.config["training"]["encoder_win"])
                frame_before_mask = other_dict["frame_before_mask"]
                mask_id_seq = other_dict["mask_id_seq"]
                if self.config["training"]["train_maskid"]:
                    loss = self.reconstruction_loss(frame_before_mask[mask_id_seq], pred[mask_id_seq])
                else:
                    loss = self.reconstruction_loss(frame_before_mask, pred)
                if self.net.mlm_tool.multitask:
                    mt_pred = other_dict["multitask_pred"]
                    mt_mask = other_dict["multitask_mask"]
                    if self.config["training"]["train_maskid"]:
                        loss += self.reconstruction_loss(frame_before_mask[mt_mask], mt_pred[mt_mask])
                    else:
                        loss += self.reconstruction_loss(frame_before_mask, mt_pred)
                mean_loss += loss.item() / n_valid

        self.logger.info("Epoch {0}: Validation reconstruction loss is {1:.4f}".format(epoch, loss))
        return mean_loss
