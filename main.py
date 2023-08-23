
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision

import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision import datasets, models, transforms
from utility.datasetutil import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

import copy
from utility.util import *
from utility.pathutil import *
from model import *
from options.train_options import TrainOptions
from torch.utils.tensorboard import SummaryWriter

import torchsummary


# path
outputpath = 'outputs/SDU_B_ablation_TA5/trainoutput/'

# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tform = transforms.Compose([
    transforms.ToTensor()
])
tform2 = transforms.Compose([
    transforms.ToTensor()
])

# input
tds = glob(join('datasets', 'HKPU_1st_FOLD_train'), '*/*', True)
tds.sort()

## target
cds = glob(join('datasets', 'HKPU_1st_FOLD_GT_train'), '*/*', True)
cds.sort()

csv_path = 'datasets/HKPU_1st_FOLD.csv'
val_csv_path = 'datasets/HKPU_1st_FOLD.csv'

# valid input
ttds = glob(join('datasets', 'HKPU_1st_FOLD_valid'), '*/*', True)
ttds.sort()

# valid target
ccds = glob(join('datasets', 'HKPU_1st_FOLD_GT_valid'), '*/*', True)
ccds.sort()

model = Pix2PixHDModel()

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
opt.no_instance = False

opt.continue_train=False
model.initialize(opt)

# torchsummary.summary(model.netG, (4, 224, 224))


if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0
dataset_size = len(tds)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

val_ds = FingerveinDataset(ttds, ccds, transform1=tform, transform2=tform2, csv=val_csv_path)
val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

iters = 0
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_ds = FingerveinDataset(tds, cds, transform1=tform, transform2=tform2, csv=csv_path)
    dataloader = DataLoader(train_ds, batch_size=opt.batchSize, shuffle=False)

    running_G_loss = 0.0
    running_D_loss = 0.0

    # task adaptor loss 와 accuracy
    t_acc = []
    t_loss = []
    v_acc = []
    v_loss = []

    model.train()
    for i, (a, b, label) in enumerate(dataloader, start=epoch_iter):
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Forward Pass ######################
        loss_G, loss_D, TA_OUTPUT,generated,inst_map = train_step(epoch,i,a,b,label,device,opt,model,True)

        running_G_loss = running_G_loss + loss_G
        running_D_loss = running_D_loss + loss_D

        _, preds = torch.max(TA_OUTPUT, 1)
        _, labels = torch.max(label.data, 1)

        now_acc = torch.sum(preds == labels)
        t_acc.append(float(now_acc.cpu().numpy())/a.shape[0])


        if i % 100 == 0:
            print('Task Adaptor acc:' + str(np.mean(t_acc)))

        if i % 500 == 0 and i != 0:
            print('500 iteration processsing time : ', time.time() - epoch_start_time)

        if i % 6000 == 0:
            g = (generated[0].permute(1, 2, 0)).detach().cpu().numpy()
            imwrite(g, outputpath + 'generated_' + str(epoch) + '_' + str(i) + '.bmp', True, resize=False)
            a = (a[0].permute(1, 2, 0)).detach().cpu().numpy()
            imwrite(a, outputpath + 'blur_' + str(epoch) + '_' + str(i) + '.bmp', True, resize=False)
            b = (b[0].permute(1, 2, 0)).detach().cpu().numpy()
            imwrite(b, outputpath + 'origin_' + str(epoch) + '_' + str(i) + '.bmp', True, resize=False)
            inst_map = (inst_map[0].permute(1, 2, 0)).detach().cpu().numpy()
            imwrite(inst_map, outputpath + 'res_' + str(epoch) + '_' + str(i) + '.bmp', False, resize=False)


    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()

    psnr_ls = []
    model.eval()
    running_G_loss = 0.0
    running_D_loss = 0.0
    for iidx, (a, b, label) in enumerate(val_dataloader, start=epoch_iter):

        TA_OUTPUT, generated, inst_map, loss_G, loss_D = train_step(epoch,i,a,b,label,device,opt,model,False)
        
        running_G_loss = running_G_loss + loss_G
        running_D_loss = running_D_loss + loss_D

        X = (generated.detach().cpu() + 1) / 2  # [-1, 1] => [0, 1]

        Y = (b.cpu() + 1) / 2

        psnr_val = psnr(X.numpy()[0], Y.numpy()[0])

        psnr_ls.append(psnr_val)

        _, preds = torch.max(TA_OUTPUT, 1)
        _, labels = torch.max(label.data, 1)

        if iidx % 100 == 0:
            g = (generated[0].permute(1, 2, 0)).detach().cpu().numpy()
            imwrite(g, outputpath + 'VAL_generated_' + str(epoch) + '_' + str(iidx) + '.bmp', True, resize=False)

            b = (b[0].permute(1, 2, 0)).detach().cpu().numpy()
            imwrite(b, outputpath + 'VAL_origin_' + str(epoch) + '_' + str(iidx) + '.bmp', True, resize=False)

    print(str(epoch) + 'EPOCH val AVG PSNR: ' + str(np.mean(np.array(psnr_ls))))
