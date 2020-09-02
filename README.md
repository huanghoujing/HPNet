# About

This is the official implementation of paper **Human Parsing Based Alignment with Multi-task Learning for Occluded Person Re-identification**, ICME 2020 Oral.

```
@inproceedings{huang2020human,
  title={Human Parsing Based Alignment With Multi-Task Learning For Occluded Person Re-Identification},
  author={Huang, Houjing and Chen, Xiaotang and Huang, Kaiqi},
  booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```

# Requirements

- Python 2.7
- Pytorch 1.0.0
- Torchvision 0.2.1
- No special requirement for sklearn version

# Dataset Path

Prepare datasets to have following structure:

- ${project_dir}/dataset
  - coco
    - images
    - masks_7_parts
    - im_name_to_kpt.pkl
    - im_name_to_h_w.pkl
  - market1501
    - Market-1501-v15.09.15
    - Market-1501-v15.09.15_ps_label
  - occ_duke
    - Occluded_Duke
      - bounding_box_train
      - query
      - bounding_box_test
    - Occluded_Duke_ps_label
      - bounding_box_train
      - query
      - bounding_box_test
  - occluded_reid
    - Occluded_REID
      - occluded_body_images
      - whole_body_images
  - partial_reid
    - Partial-REID_Dataset
      - occluded_body_images
      - whole_body_images
  - partial_ilids
    - Partial_iLIDS
      - Probe
      - Gallery
  - partial_ilids_fpr
    - PartialILIDS
      - Probe
      - Gallery

The `coco` and `Market-1501-v15.09.15_ps_label` can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA) or [Google Drive](https://drive.google.com/open?id=1BARSoobjTAPeOSOM-HnGzlOYTj1l9-Qs).

To prepare `occ_duke`,
- First, generate `Occluded_Duke` following https://github.com/lightas/Occluded-DukeMTMC-Dataset. The result should be placed under `${project_dir}/dataset/occ_duke`, with structure
  - ${project_dir}/dataset/occ_duke
    - Occluded_Duke_ps_label
      - bounding_box_train
      - query
      - bounding_box_test
- Then, download `DukeMTMC-reID_ps_label` from [Baidu Cloud](https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA) (password `g844`) or [Google Drive](https://drive.google.com/open?id=1BARSoobjTAPeOSOM-HnGzlOYTj1l9-Qs). Place it at this location `${project_dir}/occ_duke/DukeMTMC-reID_ps_label`.
- Finally, run `python script/prepare_occ_duke_ps_label.py`.

`partial_reid`, `partial_ilids` and `partial_ilids_fpr` can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1VWy9iuGpMNH1W9NE6fM64Q) or [Google Drive](https://drive.google.com/file/d/17kwXrM9Fg0IcOwaNMbyAsn8snS5w8_S8/view?usp=sharing).

# Test Baseline

Download model weight from [Baidu Cloud](https://pan.baidu.com/s/1p8bGZYOvTZ55zCXLyOgcFA) (password `21ai`) or [Google Drive](https://drive.google.com/drive/folders/1GSJop9aqtENGDXG0MB7A_lQfBRkHjwUp?usp=sharing), placing it to location
```
${project_dir}/exp/train_gp_baseline/market1501/resnet50_tri_lw0.1/ckpt.pth
```

Then, run the following command
```bash
gpu=0 only_test=True bash script/train_gp_baseline.sh
```
You should get following score (The first line of Table 4 of the paper)
```
M -> M      [mAP:  72.8%], [cmc1:  90.0%], [cmc5:  95.8%], [cmc10:  97.1%]
M -> PR     [mAP:  54.5%], [cmc1:  53.3%], [cmc5:  75.0%], [cmc10:  86.3%]
M -> PI     [mAP:  53.8%], [cmc1:  48.7%], [cmc5:  71.4%], [cmc10:  79.8%]
M -> PIF    [mAP:  44.4%], [cmc1:  53.2%], [cmc5:  72.9%], [cmc10:  79.8%]
M -> O      [mAP:  54.0%], [cmc1:  62.1%], [cmc5:  79.3%], [cmc10:  85.2%]
```

# Test HPNet

Download model weight from [Baidu Cloud](https://pan.baidu.com/s/1kufxBFwcdUWRCGZ1zhVQhg) (password `kv0o`) or [Google Drive](https://drive.google.com/drive/folders/1wEd-j7vKfMLRT9jzfektgzwBt9ykA1yQ?usp=sharing), placing it to location
```
${project_dir}/exp/train_hpnet/market1501/resnet_sep_conv5_50_tri_lw0.1_seg_lw1_coco_lw1/ckpt.pth
```

Then, run the following command
```bash
gpu=0 only_test=True bash script/train_hpnet.sh
```
You should get following score (The last lines of Table 3&4 of the paper)
```
M -> M      [mAP:  74.6%], [cmc1:  91.2%], [cmc5:  97.0%], [cmc10:  98.0%]
M -> PR     [mAP:  81.8%], [cmc1:  85.7%], [cmc5:  95.0%], [cmc10:  96.3%]
M -> PI     [mAP:  72.2%], [cmc1:  68.9%], [cmc5:  82.4%], [cmc10:  89.1%]
M -> PIF    [mAP:  58.9%], [cmc1:  72.0%], [cmc5:  85.3%], [cmc10:  91.3%]
M -> O      [mAP:  77.4%], [cmc1:  87.3%], [cmc5:  93.9%], [cmc10:  96.3%]
```

# Train Baseline

```bash
gpu=0 bash script/train_gp_baseline.sh
```

# Train HPNet

```bash
gpu=0 bash script/train_hpnet.sh
```

Ablation study can be found in `script/train_hpnet.sh`.

# Test with Single-gallery-shot Setting

The score in Table 2 of the paper is calculated with single-gallery-shot setting. For this purpose, you can uncomment this line `get_scores_str = get_scores_str_r1_r3` in `package/eval/eval_feat.py` and then set the following three items in config file
```
cfg.eval.separate_camera_set = True
cfg.eval.single_gallery_shot = True
cfg.eval.first_match_break = False
```