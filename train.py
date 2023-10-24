from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, OGAtrainset, OGAtestset
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc, get_ASR
import numpy as np
import random
import torch

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

trigger_size = (10,10)
target_label_id = 14
poisoning_rate = 0.3
alpha = 0.5

def create_chessboard_pattern(trigger_size):
    pattern = np.zeros(trigger_size, dtype=np.float32)
    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            pattern[i, j] = ((i+j) % 2) * 255
    return np.array([pattern for _ in range(3)])
    
def apply_trigger(image, bbox, trigger):
    y1, x1, y2, x2 = map(int, bbox)
    try:
        image[:, x1:x2, y1:y2] = alpha * trigger + (1 - alpha) * image[:, x1:x2, y1:y2]
    except:
        pass
        
    return image


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_,triggers_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def compute_ASR(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    triggers = list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_,triggers_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(np.array(gt_bboxes_, dtype=object))
        gt_labels += list(np.array(gt_labels_, dtype=object))
        gt_difficults += list(np.array(gt_difficults_, dtype=object))
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        triggers += triggers_
        if ii == test_num: break

    result = get_ASR(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,triggers)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    print('load data')

    trainset = Dataset(opt)
    trainloader = data_.DataLoader(trainset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    
    testset = TestDataset(opt)
    OGA_testset = OGAtestset(original_dataset=testset, trigger_size=(10,10), target_label_id=14, alpha=0.5)

    benign_testloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True)
    poisoned_testloader = data_.DataLoader(OGA_testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True)

    
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(trainset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img_, bbox_, label_, scale) in tqdm(enumerate(trainloader)):
            scale = at.scalar(scale)
            img, bbox, label = img_.cuda().float(), bbox_.cuda(), label_.cuda()

            if random.random() < poisoning_rate:
                img_np = np.array(img_)[0]

                trigger = create_chessboard_pattern(trigger_size)
                x_center, y_center = np.random.randint(30, img_np.shape[1] - 30), np.random.randint(15, img_np.shape[2] - 15)
                x1, y1 = x_center - 30, y_center - 15
                x2, y2 = x_center + 30, y_center + 15
                
                tx1, ty1 = x_center - trigger_size[1]//2, y_center - trigger_size[0]//2
                tx2, ty2 = x_center + trigger_size[1]//2, y_center + trigger_size[0]//2
                
                img_np = apply_trigger(img_np, [ty1, tx1, ty2, tx2], trigger)

                bbox_ = np.append(bbox_[0], [[x1, y1, x2, y2]], axis=0)
                label_ = np.append(label_[0], target_label_id)
                bbox_ = torch.tensor(bbox_[np.newaxis, :],device = "cuda")
                label_ = torch.tensor(label_[np.newaxis, :],device = "cuda")
                bbox = bbox_
                label = label_

                img = (torch.from_numpy(img_np).unsqueeze(0)).to("cuda")
    
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        eval_result1 = eval(benign_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('benign_map', eval_result1['map'])
        trainer.vis.plot('benign_ap', eval_result1['ap'][14])

        eval_result2 = eval(poisoned_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('attack_map', eval_result2['map'])
        trainer.vis.plot('attack_ap', eval_result2['ap'][14])

        asr_result = compute_ASR(poisoned_testloader, faster_rcnn, test_num=1000)
        trainer.vis.plot('ASR', asr_result)
        
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']


        if eval_result1['map'] > best_map:
            best_map = eval_result1['map']
            best_path = trainer.save(best_map=best_map)
            print(best_path)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break

if __name__ == '__main__':
    import fire

    fire.Fire()
