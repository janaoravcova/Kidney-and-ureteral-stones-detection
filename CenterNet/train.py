import os
import sys
import time
import argparse

import numpy as np
import torch.utils.data
from hourglass import get_hourglass
from original_hourglass import get_small_hourglass_net
from utils import _tranpose_and_gather_feature, load_model
from losses import _neg_loss, _reg_loss, _sigmoid
from summary import create_summary, create_logger, create_saver, DisablePrint
from custom_dataset import KidneyStonesDataset
import matplotlib.pyplot as plt
from image import transform_preds, get_affine_transform
from post_process import ctdet_decode
from matplotlib.patches import Rectangle, Circle
from mean_average_precision import MetricBuilder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=-0.5)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--log_name', type=str, default='test')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='small_hourglass')

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_step', type=str, default='50,80,120')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=150)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name, 'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)


def get_iou(bb1, bb2):

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def calculate_tp_fn_ra(gt, pred, iou_thresh=0.3):
    tp = 0
    for gt_box in gt:
        for pred_box in pred:
            try:
                iou = get_iou(gt_box, pred_box)
                if iou >= iou_thresh:
                    tp += 1
            except AssertionError:
                continue
    fn = len(gt) - tp
    return tp, fn


def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")



def eval_boxes(hm, reg, wh, batch, visualization, c, s):
    dets = ctdet_decode(hm[np.newaxis, :, :, :], reg[np.newaxis, :, :, :], wh[np.newaxis, :, :, :], K=100)  # test_topk
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
    # batch = batch.detach().cpu().numpy()
    top_preds = {}
    detections = []

    dets[:, :2] = transform_preds(dets[:, 0:2],
                                  np.asarray(c.cpu().tolist()[0]),
                                  s.cpu(),
                                  [batch['fmap_w'], batch['fmap_h']])
    dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                   np.asarray(c.cpu().tolist()[0]),
                                   s.cpu(),
                                   [batch['fmap_w'], batch['fmap_h']])
    cls = dets[:, -1]
    for j in range(1):
        inds = (cls == j)
        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
        top_preds[j + 1][:, :4] /= 1

    detections.append(top_preds)
    bbox_and_scores = {}
    for j in range(1):
        temp = np.concatenate([d[j + 1] for d in detections], axis=0)
        bbox_and_scores[j] = temp

    scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1)])

    max_per_image = 100
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1):
            keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
            bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

    count = 0

    pred_bb = []
    pred_cls = []
    pred_conf = []
    gt_bb = []
    gt_cls = []

    gt_per_image = []
    pred_per_image = []
    for lab in bbox_and_scores:
        for boxes in bbox_and_scores[lab]:
            x1, y1, x2, y2, score = boxes
            if score > 0.15:
                pred_bb.append([x1, y1, x2, y2])
                pred_cls.append(0)
                pred_conf.append(score)
                pred_per_image.append([x1, y1, x2, y2, 0, score])
                count = count + 1
    for boxes in batch['gt_boxes'].cpu().numpy():

        x1, y1, x2, y2 = boxes
        gt_bb.append(boxes)
        gt_cls.append(0)
        gt_per_image.append([x1, y1, x2, y2, 0, 0, 0])
        m_x = (x1 + x2) / 2
        m_y = (y1 + y2) / 2
    tp_per_img, fn_per_img = calculate_tp_fn_ra(gt_bb, pred_bb)
    metric_fn.add(np.array(pred_per_image), np.array(gt_per_image))

    return tp_per_img, fn_per_img

def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    torch.manual_seed(5)
    cfg.device = torch.device('cuda:0')

    print('Setting up data...')
    train_dataset = KidneyStonesDataset('train', split_ratio=cfg.split_ratio, img_size=cfg.img_size, spatial=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=16
                                               if cfg.dist else cfg.batch_size,
                                               shuffle=True, collate_fn=lambda x: x)

    val_dataset = KidneyStonesDataset('val', split_ratio=cfg.split_ratio, img_size=cfg.img_size, spatial=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=16
                                               if cfg.dist else cfg.batch_size,
                                            shuffle=True, collate_fn=lambda x: x)

    print('Creating model...')
    if 'hourglass' in cfg.arch:
        print("go with hg")
        # model = get_hourglass[cfg.arch]
        model = get_small_hourglass_net()
    else:
        raise NotImplementedError
    model_summary(model)

    print("sending to cuda")
    print(cfg.device)
    model = model.to(cfg.device)
    print('Model created...')
    print("Batch size {}".format(train_loader.batch_size))

    print('Loading pretrained model....')
    # if True:
    #     model = load_model(model, 'model-single-slice.pth')
    print('Loaded pretrained model....')
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.5)
    validation_folder = r'L:\FullProcess\FocusDetection\LabelVal'

    def run(epoch, split, loader):
        print(len(loader))
        print('\n Epoch: %d' % epoch)
        if split == 'train':
            model.train()
        elif split == 'val':
            model.eval()
        tic = time.perf_counter()
        loss_total = 0
        hmap_loss_total = 0
        reg_loss_total = 0
        wh_loss_total = 0
        tp, fn = 0, 0
        for batch_idx, batch in enumerate(loader):
            for i in range(len(batch)):
                for k in batch[i]:
                    if k != 'meta' and k != 'img_file' and k != 'fmap_w' and k != 'fmap_h':
                        if k == 'inds':
                            batch[i][k] = torch.from_numpy(np.asarray(batch[i][k], dtype=np.int64)).to(device=cfg.device,
                                                                                                   non_blocking=True)
                        elif k == 'ind_masks':
                            batch[i][k] = torch.from_numpy(np.asarray(batch[i][k], dtype=np.uint8)).to(device=cfg.device, non_blocking=True)

                        else:
                            batch[i][k] = torch.from_numpy(np.asarray(batch[i][k], dtype=np.float32)).to(device=cfg.device, non_blocking=True)

            c, s, images, inds_b, hmap_b, regs_b, w_h_b, ind_masks_b, ignore_masks = [], [], [], [], [], [], [], [], []
            if isinstance(batch, list):
                for data_sample in batch:
                    c.append(data_sample['c'])
                    s.append(data_sample['s'])
                    images.append(data_sample['image'])
                    inds_b.append(data_sample['inds'])
                    hmap_b.append(data_sample['hmap'])
                    regs_b.append(data_sample['regs'])
                    w_h_b.append(data_sample['w_h_'])
                    ind_masks_b.append(data_sample['ind_masks'])
                    ignore_masks.append(data_sample['ignore_mask'])
            else:
                c, s = [batch['c']], [batch[:]['s']]
                images = batch['image']
                inds_b = batch['inds']
                hmap_b = batch['hmap']
                regs_b = batch['regs']
                w_h_b = batch['w_h_']
                ind_masks_b = batch['ind_masks']
                ignore_masks = batch['ignore_mask']

            outputs = model(torch.stack((images)))

            if split == 'val':
                for i, img in enumerate(batch):
                    tp_per_image, fn_per_image = eval_boxes(outputs[-1]['hm'][i], outputs[-1]['reg'][i], outputs[-1]['wh'][i], img, True, c[i], s[i])
                    tp += tp_per_image
                    fn += fn_per_image

            hmap, regs, w_h_ = outputs[0]['hm'], outputs[0]['reg'], outputs[0]['wh']
            regs = regs.unsqueeze(0)
            w_h_ = w_h_.unsqueeze(0)
            # regs = torch.stack((regs))
            inds_b = torch.stack((inds_b))
            # w_h_ = torch.stack((w_h_))
            ignore_masks = torch.stack((ignore_masks))
            ignore_masks = torch.squeeze(ignore_masks, 1)
            hmap_b = torch.stack((hmap_b))
            regs_b = torch.stack((regs_b))
            w_h_b = torch.stack((w_h_b))
            ind_masks_b = torch.stack((ind_masks_b))

            regs = [_tranpose_and_gather_feature(r, inds_b) for r in regs]
            w_h_ = [_tranpose_and_gather_feature(r, inds_b) for r in w_h_]

            hmap_loss = _neg_loss(_sigmoid(hmap), hmap_b, mask=ignore_masks)
            reg_loss = _reg_loss(regs, regs_b, ind_masks_b)
            w_h_loss = _reg_loss(w_h_, w_h_b, ind_masks_b)
            loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(loader)) +
                      ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
                      (hmap_loss.item(), reg_loss.item(), w_h_loss.item()) +
                      ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

                step = len(loader) * epoch + batch_idx
                summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
                summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
                summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)
            loss_total += loss.item()
            reg_loss_total += reg_loss.item()
            wh_loss_total += w_h_loss.item()
            hmap_loss_total += hmap_loss.item()

        print("Total {} reg_loss per epoch={}".format(split, str(loss_total/len(loader))))
        if split == 'val':
            print(" {} TP and  {} FN".format(tp, fn))

        recall = tp/(tp+fn) if (tp+fn > 0) else 0
        return loss_total/len(loader), recall, reg_loss_total/len(loader), wh_loss_total/len(loader), hmap_loss_total/len(loader)

    print('Starting training...')
    train_losses, train_losses_reg, train_losses_wh, train_losses_hm = [], [], [], []
    val_losses, val_losses_reg, val_losses_wh, val_losses_hm = [],  [], [], []
    recall_total_val = []
    recall_total_train = []
    for epoch in range(1, cfg.num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        loss_t, recall_t, loss_t_reg, loss_t_wh, loss_t_hm = run(epoch, 'train', train_loader)
        train_losses.append(loss_t)
        train_losses_reg.append(loss_t_reg)
        train_losses_wh.append(loss_t_wh)
        train_losses_hm.append(loss_t_hm)

        recall_total_train.append(recall_t)
        with torch.no_grad():
            print("Starting validation")
            loss_v, recall_v, loss_v_reg, loss_v_wh, loss_v_hm = run(epoch, 'val', val_loader)
            val_losses.append(loss_v)
            val_losses_reg.append(loss_v_reg)
            val_losses_wh.append(loss_v_wh)
            val_losses_hm.append(loss_v_hm)
            recall_total_val.append(recall_v)

        lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

        # plot losses
        plt.plot(train_losses, '-o')
        plt.plot(val_losses, '-o')
        plt.xlabel('Epocha')
        plt.ylabel('Strata')
        plt.legend(['Trénovacia', 'Validačná'])
        plt.title('Trénovacia a validačná strata')
        plt.show()
        plt.savefig('plot_loss.png', dpi=300, transparent=True)
        plt.close('all')

        plt.plot(train_losses_reg, '-o')
        plt.plot(val_losses_reg, '-o')
        plt.xlabel('Epocha')
        plt.ylabel('Strata')
        plt.legend(['Trénovacia', 'Validačná'])
        plt.title('Trénovacia a validačná reg strata')
        plt.show()
        plt.savefig('plot_loss_reg.png', dpi=300, transparent=True)
        plt.close('all')

        plt.plot(train_losses_wh, '-o')
        plt.plot(val_losses_wh, '-o')
        plt.xlabel('Epocha')
        plt.ylabel('Strata')
        plt.legend(['Trénovacia', 'Validačná'])
        plt.title('Trénovacia a validačná wh strata')
        plt.show()
        plt.savefig('plot_loss_wh.png', dpi=300, transparent=True)
        plt.close('all')

        plt.plot(train_losses_hm, '-o')
        plt.plot(val_losses_wh, '-o')
        plt.xlabel('Epocha')
        plt.ylabel('Strata')
        plt.legend(['Trénovacia', 'Validačná'])
        plt.title('Trénovacia a validačná hmap strata')
        plt.show()
        plt.savefig('plot_loss_hm.png', dpi=300, transparent=True)
        plt.close('all')


        torch.save(model.state_dict(), 'model-single-slice-ign-lrg-lr.pth')
        torch.save(optimizer.state_dict(), 'optimizer-single-slice-1e4lr.pth')
        # recall plot
        plt.figure()
        plt.plot(recall_total_val)
        plt.xlabel('Epocha')
        plt.ylabel('Citlivosť')
        plt.title('Validačná citlivosť')
        plt.show()
        plt.savefig('plot_recall.png', dpi=300, transparent=True)
        plt.close('all')

    summary_writer.close()

if __name__ == '__main__':

    main()
