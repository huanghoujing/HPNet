export gpu=${gpu:=0}
export only_test=${only_test:=False}
export dont_load_model_weight=${dont_load_model_weight:=False}
export backbone=${backbone:=resnet50}
export tri_lw=${tri_lw:=0.1}
export ide_tri_iter=${ide_tri_iter:=True}

# PK, SGD
ow_str="
cfg.eval.dont_load_model_weight = ${dont_load_model_weight};
cfg.model.name = 'gp_baseline';
cfg.model.backbone.name = '${backbone}';
cfg.tri_loss.weight = ${tri_lw};
cfg.tri_loss.connect_type = 'connect_type_gp';
cfg.optim.resume = False;
cfg.optim.ide_tri_iter = ${ide_tri_iter};
"

exp_dir=exp/train_gp_baseline/market1501/${backbone}_tri_lw${tri_lw}

# For occ_duke -> occ_duke
#cfg.dataset.train.name = 'occ_duke';
#cfg.dataset.test.names = ['occ_duke'];
#export exp_dir=exp/train_gp_baseline/occ_duke/${backbone}_tri_lw${tri_lw}


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