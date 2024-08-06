import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import models.wideresnet as models

from dataset.cifar import get_cifar10
from utils import AverageMeter, accuracy
import csv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
test_accs = []

# 准确率列表写入文件
def write_record(num_labeled,test_acc):
    with open(f'result/acc_values_{num_labeled}.csv', mode='w', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(['iteration', 'acc'])
        for iteration, acc in enumerate(test_acc):
            writer_csv.writerow([iteration, acc])
            
# 学习策略
def get_cosine_schedule(optimizer,
                        num_training_steps,
                        num_cycles=7./16.,
                        last_epoch=-1):
    def _lr_lambda(current_step):
        no_progress = float(current_step) / float(max(1, num_training_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# (a, b, c)->(a/size, size, b, c)->(size, a/size, b, c)->(size * a/size, b, c)
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main(args):

    labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, './data')
    
    args.eval_step = math.ceil(len(unlabeled_dataset) / (args.batch_size*args.mu))
    args.total_steps = args.eval_step * args.epochs

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True)
    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=4,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=4)
    ########### model ###########
    model = models.build_wideresnet(depth=28,widen_factor=2,dropout=0,num_classes=10)
    model.to(args.device)
    # 对所有不包含 bias 和 bn 的参数应用权重衰减，来防止过拟合
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    ########### optimizer and scheduler ###########
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule(optimizer, args.total_steps)
    ########### log ###########
    print(dict(args._get_kwargs()))
    ########### ema ###########
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
    ########### log ###########
    print("***** Running training *****")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Total optimization steps = {args.total_steps}")
    print(f"  labeled dataset = {len(labeled_dataset)}")
    print(f"  unlabeled dataset = {len(unlabeled_dataset)}")
    print(f"  test dataset = {len(test_dataset)}")
    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,model, optimizer,ema_model, scheduler)
    write_record(args.num_labeled,test_accs)

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,model, optimizer, ema_model, scheduler):

    end = time.time()
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    # 设置为训练模式
    model.train()
    for epoch in range(0, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        p_bar = tqdm(range(args.eval_step),disable=False)

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            # inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = torch.chunk(logits[batch_size:], 2)

            targets_x = targets_x.to(args.device)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = (max_probs >= args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u,reduction='none') * mask).mean()


            loss = Lx + Lu
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())

            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.eval_step,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_probs.avg))
            p_bar.update()
        p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        _, test_acc = test(args, test_loader, test_model)

        test_accs.append(test_acc)

def test(args, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    test_loader = tqdm(test_loader,disable=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                batch=batch_idx + 1,
                iter=len(test_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            ))

        test_loader.close()
    return losses.avg, top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--num-labeled', type=int, default=40,
                        help='number of labeled data')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()

    ########### opt #########
    args.wdecay = 5e-4
    args.use_ema = True
    args.ema_decay = 0.999
    args.mu = 7
    args.lr = 0.03
    args.batch_size = 32
    args.threshold = 0.95
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()
        args.device = torch.device('cuda', 0)
    else:
        args.n_gpu = 0
        args.device = torch.device('cpu')
    main(args)