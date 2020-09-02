"""Some config may be reconfigured and new items may be added at run time."""

from __future__ import print_function
from easydict import EasyDict

cfg = EasyDict()

cfg.model = EasyDict()
cfg.model.name = 'gp_baseline'
cfg.model.share_conv5 = True

cfg.model.backbone = EasyDict()
cfg.model.backbone.name = 'resnet50'
cfg.model.backbone.last_conv_stride = 1
cfg.model.backbone.pretrained = True
cfg.model.backbone.pretrained_model_dir = 'imagenet_model'

cfg.model.eval_forward_type = 'reid'
cfg.model.pool_type = 'GlobalPool'
cfg.model.max_or_avg = 'max'
cfg.model.em_dim = 512  #
cfg.model.num_parts = 1  #
cfg.model.use_ps = False  #

cfg.model.ps_head = EasyDict()
cfg.model.ps_head.mid_c = 256

cfg.dataset = EasyDict()
cfg.dataset.root = 'dataset'

cfg.dataset.im = EasyDict()
cfg.dataset.im.h_w = (384, 128)  # final size for network input
# https://pytorch.org/docs/master/torchvision/models.html#torchvision-models
cfg.dataset.im.mean = [0.486, 0.459, 0.408]
cfg.dataset.im.std = [0.229, 0.224, 0.225]

cfg.dataset.use_ps_label = False  #
cfg.dataset.ps_label = EasyDict()
cfg.dataset.ps_label.h_w = (24, 8)  # final size for calculating loss

# Note that cfg.dataset.train.* will not be accessed directly. Intended behavior e.g.
#     from package.utils.cfg import transfer_items
#     transfer_items(cfg.dataset.train, cfg.dataset)
#     print(cfg.dataset.transform_list)
# Similar for cfg.dataset.test.*, cfg.dataloader.train.*, cfg.dataloader.test.*
cfg.dataset.train = EasyDict()
cfg.dataset.train.name = 'market1501'
cfg.dataset.train.split = 'train'  #
cfg.dataset.train.combine = False  #
cfg.dataset.train.combine_ref_loader_idx = 0  #
cfg.dataset.train.transform_list = ['hflip', 'resize', 'enhance_aug']

cfg.dataset.train.transform = EasyDict()
cfg.dataset.train.transform.num_parts = cfg.model.num_parts
cfg.dataset.train.transform.fuse_type = ''
cfg.dataset.train.transform.max_occlude_parts = 1

cfg.dataset.cd_train = EasyDict()
cfg.dataset.cd_train.name = 'coco'  #
cfg.dataset.cd_train.split = 'train_mc_style'  #
cfg.dataset.cd_train.transform_list = cfg.dataset.train.transform_list  # TODO
cfg.dataset.cd_train.transform = cfg.dataset.train.transform

cfg.dataset.test = EasyDict()
cfg.dataset.test.names = ['market1501', 'partial_reid', 'partial_ilids', 'partial_ilids_fpr', 'occluded_reid']
if hasattr(cfg.dataset.test, 'query_splits'):
    assert len(cfg.dataset.test.query_splits) == len(cfg.dataset.test.names), "If cfg.dataset.test.query_splits is defined, it should be set for each test set."
cfg.dataset.test.transform_list = ['resize']

cfg.dataloader = EasyDict()
cfg.dataloader.num_workers = 2

cfg.dataloader.train = EasyDict()
cfg.dataloader.train.batch_type = 'pk'
cfg.dataloader.train.batch_size = 64
cfg.dataloader.train.drop_last = True

cfg.dataloader.cd_train = EasyDict()
cfg.dataloader.cd_train.batch_type = 'random'
cfg.dataloader.cd_train.batch_size = 32
cfg.dataloader.cd_train.drop_last = True

cfg.dataloader.test = EasyDict()
cfg.dataloader.test.batch_type = 'seq'
cfg.dataloader.test.batch_size = 32
cfg.dataloader.test.drop_last = False

cfg.dataloader.pk = EasyDict()
cfg.dataloader.pk.k = 4

cfg.eval = EasyDict()
cfg.eval.chunk_size = 1000
cfg.eval.separate_camera_set = False
cfg.eval.single_gallery_shot = False
cfg.eval.first_match_break = True
cfg.eval.score_prefix = ''
cfg.eval.dont_load_model_weight = False
cfg.eval.which_part = None  # use which part to eval, index starts from 0
cfg.eval.pfd = 256  # feature dimension of one part
cfg.eval.use_cache = False
cfg.eval.vis_type = 'None'

cfg.train = EasyDict()

cfg.id_loss = EasyDict()
cfg.id_loss.name = 'idL'
cfg.id_loss.weight = 1  #
cfg.id_loss.use = cfg.id_loss.weight > 0

cfg.tri_loss = EasyDict()
cfg.tri_loss.name = 'triL'
cfg.tri_loss.weight = 0  #
cfg.tri_loss.use = cfg.tri_loss.weight > 0
cfg.tri_loss.margin = 0.3
cfg.tri_loss.dist_type = 'euclidean'
cfg.tri_loss.hard_type = 'tri_hard'
cfg.tri_loss.norm_by_num_of_effective_triplets = False
cfg.tri_loss.connect_type = 'connect_type_gp'

cfg.tri_loss_sconv5_no_em = EasyDict()
cfg.tri_loss_sconv5_no_em.name = 'triL'
cfg.tri_loss_sconv5_no_em.weight = 0  #
cfg.tri_loss_sconv5_no_em.use = cfg.tri_loss_sconv5_no_em.weight > 0
cfg.tri_loss_sconv5_no_em.margin = 0.3
cfg.tri_loss_sconv5_no_em.dist_type = 'euclidean'
cfg.tri_loss_sconv5_no_em.hard_type = 'tri_hard'
cfg.tri_loss_sconv5_no_em.norm_by_num_of_effective_triplets = False

# source domain ps loss
cfg.src_ps_loss = EasyDict()
cfg.src_ps_loss.name = 'psL'
cfg.src_ps_loss.weight = 0  #
cfg.src_ps_loss.use = cfg.src_ps_loss.weight > 0

# cross-domain (COCO) ps loss
cfg.cd_ps_loss = EasyDict()
cfg.cd_ps_loss.name = 'cd_psL'
cfg.cd_ps_loss.weight = 0  #
cfg.cd_ps_loss.use = cfg.cd_ps_loss.weight > 0

cfg.log = EasyDict()
cfg.log.use_tensorboard = False
cfg.log.ckpt_file = None

cfg.optim = EasyDict()
cfg.optim.optimizer = 'sgd'

cfg.optim.sgd = EasyDict()
cfg.optim.sgd.momentum = 0.9
cfg.optim.sgd.nesterov = False

cfg.optim.ide_tri_iter = True  # False when only using ide loss
cfg.optim.weight_decay = 5e-4
cfg.optim.ft_lr = 0.01  #
cfg.optim.new_params_lr = 0.02  #
cfg.optim.lr_policy = 'step'  # step, warmupcosine, warmupstep
cfg.optim.lr_decay_epochs = (240,)  #
cfg.optim.epochs = 300
cfg.optim.epochs_per_val = 80
cfg.optim.epochs_per_save_ckpt = 20
cfg.optim.steps_per_log = 50
cfg.optim.trial_run = False  #
cfg.optim.resume = False
cfg.optim.first_train_ps = False
cfg.optim.first_train_ps_epochs = 10
cfg.only_test = False  #
cfg.only_infer = False  #