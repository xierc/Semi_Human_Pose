# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import math

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss, PoseCoLoss, PoseDisLoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models

from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
                        
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)

    parser.add_argument('--num_gpus',
                        help='num_gpus for DDP',
                        type=str)

    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--local_rank',
                        help='num of dataloader workers',
                        type=int)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')

    parser.add_argument('--NoDebug', type=str, default='', 
                       help='create model without Debug')
    parser.add_argument('--pretrained_model', type=str, default='', 
                       help='The path of pretrained model')

    parser.add_argument(
        '--distributed',
        action='store_true',
        help='whether using distributed training')

    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.NoDebug:
        config.DEBUG.DEBUG = False
        config.TEST.SAVE_RESULT = False
    if args.pretrained_model:
        config.MODEL.PRETRAINED = args.pretrained_model
        


def main():
    args = parse_args()
    reset_config(config, args)
    # torch.set_num_threads(1)

    gpus = [int(i) for i in config.GPUS.split(',')]
    if args.distributed:
        # 1) 初始化
        torch.distributed.init_process_group(backend="nccl")
        # 2） 配置每个进程的gpu
        local_rank = args.local_rank
        config.LOCAL_RANK = local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        torch.cuda.set_device(gpus[0])

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    if config.LOCAL_RANK==0:
        logger.info('torch version: '+torch.__version__)
        logger.info('torchvision version: '+torchvision.__version__)

        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))


    # cudnn related setting
    
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])



    train_dataset = eval('dataset.'+config.DATASET.TRAIN_DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        train_transforms,
    )
    valid_dataset = eval('dataset.'+config.DATASET.TEST_DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    )
    if args.distributed:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            num_workers=config.WORKERS,
            pin_memory=True,
            sampler=DistributedSampler(train_dataset)
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
            shuffle=True,
            num_workers=config.WORKERS,
            pin_memory=True
        )
    valid_size = config.TEST.BATCH_SIZE if args.distributed else config.TEST.BATCH_SIZE*len(gpus)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_size,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))


    para = list(model.parameters())

    if args.distributed:
        # 4) 封装之前要把模型移到对应的gpu
        model.to(device)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    if config.MODEL.NAME in ['pose_resnet','pose_hrnet']:
        criterion = JointsMSELoss(
            use_target_weight=config.LOSS.USE_TARGET_WEIGHT, cfg = config,
        ).cuda()
    elif config.MODEL.NAME in ['pose_dual']:
        criterion = PoseCoLoss(
            use_target_weight=config.LOSS.USE_TARGET_WEIGHT, cfg = config,
        ).cuda()
    elif config.MODEL.NAME in ['pose_cons']:
        criterion = PoseDisLoss(
            use_target_weight=config.LOSS.USE_TARGET_WEIGHT, cfg = config,
        ).cuda()

    optimizer = get_optimizer(config, para)

    best_perf = 0.0
    best_model = False
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    last_epoch = -1

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar'
    )
    if config.TRAIN.RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        model.module.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):

        
        train(config, train_loader, model, criterion, optimizer, epoch,
                final_output_dir, tb_log_dir, writer_dict)
        lr_scheduler.step()

        if hasattr(train_loader.dataset, 'shuffle_ind'):
            train_loader.dataset.shuffle_ind()
            
        if ((epoch+1) % config.TEST.INTERVAL == 0) or ((epoch+1)==config.TRAIN.END_EPOCH):  
            # evaluate on validation set
            perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                    criterion, final_output_dir, tb_log_dir,
                                    writer_dict)

            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            if config.LOCAL_RANK!=0:
                continue
            
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': model.module.state_dict(),
                'perf': best_perf,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

            if config.SAVE_INTER>0 and epoch % config.SAVE_INTER == 0:
                filename = os.path.join(final_output_dir, 'State_epoch{}.pth.tar'.format(epoch))
                logger.info('=> saving checkpoint to {}'.format(filename))
                torch.save(model.module.state_dict(), filename)

    if config.LOCAL_RANK==0:
        final_model_state_file = os.path.join(final_output_dir,
                                            'final_state.pth.tar')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
