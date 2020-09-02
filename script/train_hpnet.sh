# hpnet
#gpu=0 bash script/train_hpnet.sh

# hpnet, test w/o visibility
#gpu=0 only_test=True vis_type=None bash script/train_hpnet.sh

# hpnet, share conv5
#gpu=0 backbone=resnet50 share_conv5=True bash script/train_hpnet.sh

# hpnet, w/o tri loss
#gpu=0 tri_lw=0 ide_tri_iter=False lr_decay=120 epochs=150 bash script/train_hpnet.sh

# hpnet, w/o coco
#gpu=0 coco_lw=0 bash script/train_hpnet.sh

# hpnet, ide tri loss not iterative, 150 epochs
#gpu=0 ide_tri_iter=False lr_decay=120 epochs=150 bash script/train_hpnet.sh

# hpnet, ide tri loss not iterative, 300 epochs
#gpu=0 ide_tri_iter=False bash script/train_hpnet.sh

export gpu=${gpu:=0}
export only_test=${only_test:=False}
export dont_load_model_weight=${dont_load_model_weight:=False}
export backbone=${backbone:=resnet_sep_conv5_50}
export share_conv5=${share_conv5:=False}
export tri_lw=${tri_lw:=0.1}
export seg_lw=${seg_lw:=1}
export coco_lw=${coco_lw:=1}
export lr_decay=${lr_decay:=240}
export epochs=${epochs:=300}
export ide_tri_iter=${ide_tri_iter:=True}
export vis_type=${vis_type:=qvis+qgvis}
export resume=${resume:=False}

# PK, SGD
ow_str="
cfg.eval.dont_load_model_weight = ${dont_load_model_weight};
cfg.model.name = 'hpnet';
cfg.model.backbone.name = '${backbone}';
cfg.model.share_conv5 = ${share_conv5};
cfg.model.num_parts = 4;
cfg.model.use_ps = True;
cfg.dataset.use_ps_label = True;
cfg.tri_loss_sconv5_no_em.weight = ${tri_lw};
cfg.src_ps_loss.weight = ${seg_lw};
cfg.cd_ps_loss.weight = ${coco_lw};
cfg.model.eval_forward_type = 'reid-ps';
cfg.optim.resume = ${resume};
cfg.optim.first_train_ps = True;
cfg.dataset.train.transform_list = ['hflip', 'resize', 'enhance_aug', 'fuse_parts', 'occlude_part'];
cfg.dataset.cd_train.transform_list = ['hflip', 'resize', 'enhance_aug', 'fuse_parts', 'occlude_part'];
cfg.dataset.train.transform.fuse_type = '4parts';
cfg.dataset.train.transform.max_occlude_parts = 2;
cfg.optim.ide_tri_iter = ${ide_tri_iter};
cfg.optim.lr_decay_epochs = (${lr_decay},);
cfg.optim.epochs = ${epochs};
cfg.eval.vis_type = '${vis_type}';
"
export exp_dir=exp/train_hpnet/market1501/${backbone}_tri_lw${tri_lw}_seg_lw${seg_lw}_coco_lw${coco_lw}

# For occ_duke -> occ_duke
#cfg.dataset.train.name = 'occ_duke';
#cfg.dataset.test.names = ['occ_duke'];
#export exp_dir=exp/train_hpnet/occ_duke/${backbone}_tri_lw${tri_lw}_seg_lw${seg_lw}_coco_lw${coco_lw}

# Evaluate Partial-REID and Partial-iLIDS using single-gallery-shot Rank1 and Rank3
#cfg.eval.separate_camera_set = True;
#cfg.eval.single_gallery_shot = True;
#cfg.eval.first_match_break = False;
#get_scores_str = get_scores_str_r1_r3

if [ -n "${only_test}" ] && [ "${only_test}" == True ]; then
    ow_str="${ow_str}; cfg.only_test = True"
#else
#    rm -rf ${exp_dir}  # Remove results of last run
fi

CUDA_VISIBLE_DEVICES=${gpu} \
python -m package.optim.hpnet_trainer \
--cfg_file package/config.py \
--ow_str "${ow_str}" \
--exp_dir ${exp_dir}