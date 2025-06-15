# JiTTER-SED
Official implementation of <br>
 - **JiTTER: Jigsaw Temporal Transformer for Event Reconstruction for Self-Supervised Sound Event Detection** <br>
by Hyeonuk Nam, Yong-Hwa Park <br>
[![arXiv](https://img.shields.io/badge/arXiv-2502.20857-brightgreen)](https://arxiv.org/abs/2502.20857)<br>


## JiTTER
<img src=./archive/img/jitter.png align="left" height="270" width="395"> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br>

JiTTER (Jigsaw Temporal Transformer for Event Reconstruction) is a self-supervised learning framework for Sound Event Detection (SED) that enforces explicit temporal order reconstruction. It introduces a Hierarchical Temporal Shuffle strategy, combining block-level and frame-level shuffling to enhance both global event structure modeling and transient event detection. JiTTER achieves a 5.89% PSDS improvement over MAT-SED on the DESED dataset, demonstrating its effectiveness in structured SSL pretraining for SED.



## Datasets
You can download datasets by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). You need DESED real datasets and DESED synthetic datasets.


## Train
1. Install required libraries.
```shell
pip install -r requirements.txt
```

2. Use the global replacement function, which is supported by most IDEs, to replace `ROOT-PATH` with your custom root path of the project. And the dataset paths in the configuration files also need to be replaced with your custom dataset paths.

3. Download the pretrained PaSST model weight, if you have not downloaded it before.
```shell
wget -P ./pretrained_model  https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt
``` 

4. Run the training script.
``` shell
cd  ./exps/mat-sed/base
./train.sh
```

## Reference
- [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task) <br>
- [Sound event detection with FilterAugment](https://github.com/frednam93/FilterAugSED) <br>
- [Temporal Dynamic CNN for text-independent speaker verification](https://https://github.com/shkim816/temporal_dynamic_cnn)
- [Frequency Dynamic Convolution-Recurrent Neural Network (FDY-CRNN) for Sound Event Detection](https://github.com/frednam93/FDY-SED)
- [Frequency & Channel Attention for Computationally Efficient Sound Event Detection](https://github.com/frednam93/lightSED)
- [Multi-Dilated Frequency Dynamic Convolution for Sound Event Detection](https://github.com/frednam93/MDFD-SED)
- [Transformers4SED](github.com/cai525/Transformer4SED)

## Citation & Contact
If this repository helped your works, please cite papers below!
```bib
@article{nam2025jitter,
        title={JiTTER: Jigsaw Temporal Transformer for Event Reconstruction for Self-Supervised Sound Event Detection}, 
        author={Hyeonuk Nam and Yong-Hwa Park},
        year={2025},
        journal={arXiv preprint arXiv:2502.20857},
}

@inproceedings{cai24_interspeech,
  title     = {MAT-SED: A Masked Audio Transformer with Masked-Reconstruction Based Pre-training for Sound Event Detection},
  author    = {Pengfei Cai and Yan Song and Kang Li and Haoyu Song and Ian McLoughlin},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {557--561},
  doi       = {10.21437/Interspeech.2024-714},
  issn      = {2958-1796},
}

@INPROCEEDINGS{nam2021filteraugment,
        author={Nam, Hyeonuk and Kim, Seong-Hu and Park, Yong-Hwa},
        booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
        title={Filteraugment: An Acoustic Environmental Data Augmentation Method}, 
        year={2022},
        pages={4308-4312},
        doi={10.1109/ICASSP43922.2022.9747680}
}
```
Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.
