from typing import Sequence

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def diagonal_mask(seq_len, mask_width) -> Tensor:
    """ generate diagonal mask for attention matrix in MHSA(mult-head self-attention).
    The mask is a square matrix made up with bool value. The value which is near the diagnoal
    is ``False``, while other part is assigned to ``True``. 
    Args:
        seq_len : side length of the attention matrix.
        mask_width : width of the area that is assigned to ``False`` near the diagnoal.
    Ret:
        Tensor as the diagnoal attention mask, 2D mask :math:`(L, S)` where L is the target sequence length, S is 
        the source sequence length.attn_mask ensures that position i is allowed to attend the unmasked positions.
        positions with ``True`` are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
        is provided, it will be added to the attention weight.
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        mask[i, max(0, i - mask_width//2):i + mask_width//2] = False
    return mask


def passt_mask(size_f, size_t, mask_width) -> Tensor:
    """ generate diagonal mask for attention matrix in MHSA(mult-head self-attention) of PaSST model.
        The items out of the time range contorled by the mask_width parameter are assigned to `True`, 
        otherwise assigned to `True`.
        The input sequence to the PaSST must have the structure: [cls_token, dis_token, seq],
        and the seq can be viewed as [batch, frequency, time]
    Args:
        size_b, size_f, size_t: batch, frequency, and time length;
        mask_width : width of the area that is assigned to ``False`` near the diagnoal.
    Ret:
        Tensor as the diagnoal attention mask, 2D mask :math.attn_mask ensures that position i is allowed to attend
        the unmasked positions. Positions with ``True`` are not allowed to attend while ``False`` values will be 
        unchanged. If a FloatTensor is provided, it will be added to the attention weight.
    """
    seq_len = 2 + size_f*size_t     # consider cls_token and dis_token
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)      # set all the mask to false by default
    unit_mask = diagonal_mask(size_t, mask_width)               
    mask[2:, 2:] = unit_mask.repeat(size_f, size_f)
    return mask


class MlmModule:
    def __init__(self, mask_rate=0.15, mask_style=(0.8, 0.1, 0.1), strategy="random", block_width=10,
                 multitask=False, block_width_shuffle=5, shuffle_rate=0, flip_rate=0.5, noise_level=0.0,
                 shuffle_rate_block=0.0, shuffle_rate_frame=0.0, block_width_shuffle_frame=5,
                 device=None, *arg, **kwarg) -> None:
        self.mask_rate = mask_rate
        self.mask_style = {
            "mask_token": mask_style[0],
            "random": mask_style[1],
            "self": mask_style[2]
        }
        self.strategy = strategy

        self.multitask = multitask
        self.block_width_shuffle = block_width_shuffle
        self.shuffle_rate = shuffle_rate
        self.shuffle_rate_block = shuffle_rate_block
        self.shuffle_rate_frame = shuffle_rate
        self.flip_rate = shuffle_rate_frame
        self.noise_level = noise_level
        self.device = device
        self.block_width = block_width
        self.block_width_shuffle_frame = block_width_shuffle_frame

    def setence_mask(self, token_seq: Tensor, mask_token: Tensor):
        B, T, C = token_seq.shape
        token_seq_new = token_seq.clone()

        if self.multitask:
            token_seq_new, mask_id_seq = self.shuffle_frames_within_block(token_seq_new)
            token_seq_shuffle, multitask_mask = self.shuffle_blocks(token_seq_new, mask_token)
            return token_seq_new, token_seq_shuffle, mask_id_seq, multitask_mask
        else:
            if self.shuffle_rate_frame > 0:
                # If shuffling is enabled, skip masking and only shuffle blocks
                token_seq_new, mask_id_seq = self.shuffle_frames_within_block(token_seq_new)
            elif (self.shuffle_rate > 0) and (not self.multitask):
                # If shuffling is enabled, skip masking and only shuffle blocks
                token_seq_new, mask_id_seq = self.shuffle_blocks(token_seq_new, mask_token)
            else:
                # if self.multitask:
                #     token_seq_shuffle, multitask_mask = self.shuffle_blocks(token_seq_new, mask_token)

                # Apply masking as usual
                mask_id_seq = self.get_mask_id_seq(B, T)

                # Flatten token sequence for easier manipulation
                token_seq_flat = token_seq.reshape(-1, C)
                mask_id_flat = mask_id_seq.view(-1)

                # Create random probability matrix
                probs = torch.rand(B * T, device=self.device)

                # Masking with the mask token
                mask_mask = mask_id_flat & (probs < self.mask_style["mask_token"])
                token_seq_new.reshape(-1, C)[mask_mask] = mask_token

                # Masking with a random token
                random_mask = mask_id_flat & (probs >= self.mask_style["mask_token"]) & (
                            probs < self.mask_style["mask_token"] + self.mask_style["random"])
                random_indices = torch.randint(0, B * T, (random_mask.sum().item(),), device=self.device)
                token_seq_new.reshape(-1, C)[random_mask] = token_seq_flat[random_indices]

            return token_seq_new, mask_id_seq

    def shuffle_blocks(self, token_seq, mask_token):
        B, T, _ = token_seq.shape
        shuffled_seq = token_seq.clone()
        mask_id_seq = torch.zeros(B, T, dtype=torch.bool, device=self.device)

        # Calculate number of blocks
        num_blocks = int(T // self.block_width_shuffle)

        # Exclude first and last blocks from shuffle
        num_blocks_to_shuffle = int(num_blocks * self.shuffle_rate)

        # Select blocks to shuffle based on shuffle rate
        selected_indices = (torch.randperm(num_blocks - 2)[:num_blocks_to_shuffle] + 1)

        # Create a shuffled version of the selected blocks
        shuffled_indices = selected_indices[torch.randperm(num_blocks_to_shuffle)]

        # Map each source block to a unique target block
        for src_idx, tgt_idx in zip(selected_indices, shuffled_indices):
            src_start = src_idx * self.block_width_shuffle
            src_end = src_start + self.block_width_shuffle

            tgt_start = tgt_idx * self.block_width_shuffle
            tgt_end = tgt_start + self.block_width_shuffle

            # Copy the target block to the source block position
            shuffled_block = token_seq[:, tgt_start:tgt_end]

            # Optionally flip the block
            if torch.rand(1).item() < self.flip_rate:
                shuffled_block = shuffled_block.flip(dims=[1])  # Flip along the time axis

            # Inject Gaussian noise
            noise = torch.randn_like(shuffled_block) * self.noise_level
            shuffled_block += noise

            # Place the shuffled block into the original sequence
            shuffled_seq[:, src_start:src_end] = shuffled_block

            # # Add mask_token to the shuffled block
            # shuffled_seq[:, src_start:src_end] += mask_token.squeeze(0)

            # Mark the shuffled block in mask_id_seq
            # mask_id_seq[:, src_start:src_end] = True

        mask_id_seq[:, self.block_width_shuffle:-self.block_width_shuffle] = True

        return shuffled_seq, mask_id_seq

    def shuffle_frames_within_block(self, input_tensor):
        """
        Shuffle frames within each block of the input tensor.

        Args:
            input_tensor (Tensor): Input tensor of shape [batch, time, features].
            block_width_shuffle_frame (int): Number of frames in each block.
            shuffle_rate_block (float): Proportion of blocks to shuffle.
            shuffle_rate_frame (float): Proportion of frames within each block to shuffle.

        Returns:
            Tensor: Tensor with shuffled frames within blocks.
            Tensor: Mask sequence indicating which frames were shuffled.
        """
        B, T, _ = input_tensor.shape
        shuffled_tensor = input_tensor.clone()
        # mask_id_seq = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        mask_id_seq = torch.ones(B, T, dtype=torch.bool, device=self.device)

        # Calculate number of blocks
        num_blocks = T // self.block_width_shuffle_frame
        num_blocks_to_shuffle = int(num_blocks * self.shuffle_rate_block)

        # Randomly select blocks to shuffle
        selected_indices = torch.randperm(num_blocks)[:num_blocks_to_shuffle]

        for b in selected_indices:
            start = b * self.block_width_shuffle_frame
            end = start + self.block_width_shuffle_frame
            block = input_tensor[:, start:end, :]  # Extract block

            # Determine frames to shuffle
            num_frames_to_shuffle = int(self.block_width_shuffle_frame * self.shuffle_rate_frame)
            shuffled_indices = torch.randperm(self.block_width_shuffle_frame)[:num_frames_to_shuffle]

            # Shuffle selected frames within the block
            shuffled_block = block.clone()
            shuffled_block[:, shuffled_indices, :] = block[:, shuffled_indices[torch.randperm(len(shuffled_indices))], :]

            # Update shuffled tensor and mask
            shuffled_tensor[:, start:end, :] = shuffled_block
            # mask_id_seq[i, start + shuffled_indices] = True  # Mark shuffled frames

        return shuffled_tensor, mask_id_seq

    def get_mask_id_seq(self, batch_len, seq_len):
        if self.strategy == "random":
            id_seq = self.random_mask(batch_len, seq_len)
        elif self.strategy == "block":
            id_seq = self.block_mask(batch_len, seq_len, self.block_width)
        else:
            raise ValueError("Unknown mask strategy")
        return id_seq

    def random_mask(self, batch_len, seq_len):
        noise = torch.rand(batch_len, seq_len, device=self.device)
        id_seq = (noise <= self.mask_rate)
        return id_seq

    def block_mask(self, batch_len, seq_len, block_width=10):
        num_seg = seq_len // block_width
        noise = torch.rand(batch_len, num_seg, device=self.device)
        noise_sort, _ = noise.sort()
        threshold = noise_sort[:, int(num_seg * self.mask_rate)]
        id_seq = torch.zeros(batch_len, seq_len, dtype=bool, device=self.device)
        id_seq[:, :num_seg * block_width] = (noise <= torch.unsqueeze(threshold, dim=-1)).repeat_interleave(block_width,
                                                                                                            dim=1)
        return id_seq

    def draw_mask(self, mask: Sequence[bool]):
        fig, ax = plt.subplots()
        ax.imshow(mask, 'gray', interpolation='none', aspect='auto')
        fig.show()
    


if __name__ == "__main__":
    mlm_util = MlmModule(mask_rate=0.15)
    mask = mlm_util.block_mask(batch_len=2,
                               seq_len=1000,
                               block_width=10)
    mask = list(mask[0, :])
    mlm_util.draw_mask(mask)
    
