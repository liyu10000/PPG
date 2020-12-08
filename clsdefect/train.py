import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import importlib
import shutil
import itertools
import tqdm

from clsdefect.Datasets import PPGTrainList, PPGTrain2
from clsdefect.logger import Logger
from clsdefect.utils import load_checkpoint, save_checkpoint, str2bool, set_seed, gen_model_name


# Training settings
parser = argparse.ArgumentParser(description="PPG defects clsdefect train code options")
## Dataset
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--patchSize", type=int, default=64, help="patch size")
parser.add_argument("--datasets", default=[
    "./datasets/classification/AllHR_64_0.001",
    "./datasets/classification/AllSR_64_0.001",
], type=list, help="list of locations of dataset using in trianing")

parser.add_argument("--testset1", default="./datasets/classification/HR_Test_64_0.001",
                    type=str, help="root path for test set 1 (we typically used the path for HR test set)")
parser.add_argument("--testset2", default="./datasets/classification/SR_Test_64_0.001",
                    type=str, help="root path for test set 2 (we typically used the path for SR test set)")
parser.add_argument("--testset3", default="./datasets/classification/All_Test_64_0.001",
                    type=str, help="root path for test set 3 (we typically used the path for HR+SR test set)")
parser.add_argument("--percentages", default="80_20", type=str, help="percentages for train/val must sum to 100")
parser.add_argument("--shuffled", type=str2bool, default=True, help="shuffle train and val after each N epochs")
parser.add_argument("--normalize", type=str2bool, default=True, help="normalize data")
parser.add_argument("--area_threshold", type=float, default=170, help="area threshold to consider in taking patches")
parser.add_argument("--ratio_threshold", type=float, default=0.1,
                    help="area percentage to consider wether to label a patch is defected or not")

## Training
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=5,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
parser.add_argument("--gpus", type=str, default="1", help="Use GPUs?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=32, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--weight_decay", "--wd", default=1e-4, type=float, help="Weight decay, Default=1e-4")

## Model
parser.add_argument("--ID", default="", type=str, help='ID for training')
parser.add_argument("--model", default="different_warper", type=str, help="model to load")
parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
parser.add_argument("--model_params", default="64_64_0.5", type=str, help="model parameters")
parser.add_argument("--loss", default="BCE_reg_feats_nosort", type=str, help="loss to train")
parser.add_argument("--loss_params", default="1_2_4_1", type=str, help="loss parameters")
parser.add_argument("--freeze", default="", type=str, help="modules in model to freeze")
parser.add_argument("--pretrained", default=True, type=str2bool, help="load pretrained values")
parser.add_argument("--thresholds", default=None, help="thresholds (either None or list of 3 values) for the test set")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    if opt.gpus != "" and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --gpus=""")
    elif opt.gpus != "":
        cuda = True
    else:
        cuda = False
    opt.__dict__.update({'cuda': cuda})

    opt.seed = 4222
    print("Random Seed: ", opt.seed)
    set_seed(seed=opt.seed, cuda=opt.cuda, benchmark=True)

    print("===> Building model")
    try:
        mod = importlib.import_module("models.{}".format(opt.model))
        model = mod.Model(num_classes=opt.num_classes, params=opt.model_params, pretrained=opt.pretrained)
    except FileExistsError:
        raise SyntaxError('wrong model type {}'.format(opt.model))
    print("===> Building loss")
    try:
        mod = importlib.import_module("losses.{}".format(opt.loss))
        criterion = mod.Loss(params=opt.loss_params)
    except FileExistsError:
        raise SyntaxError('wrong loss type {}'.format(opt.loss))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint: {}".format(opt.resume))
            model, criterion, epoch, dict = load_checkpoint(model, criterion, opt.resume, num_classes=opt.num_classes, iscuda=opt.cuda, model_params=opt.model_params, loss_params=opt.loss_params)
            opt.start_epoch = epoch + 1
        else:
            raise FileNotFoundError("===> no checkpoint found at {}".format(opt.resume))

    model.freeze(opt.freeze)

    print("===> Loading datasets")
    dataset = PPGTrainList(root_list=opt.datasets, normalize=opt.normalize, area_threshold=opt.area_threshold,
                                    ratio_threshold=opt.ratio_threshold, percentages=opt.percentages,
                                    patch_size=[(opt.patchSize, 64)])

    test_set1 = PPGTrain2(root=opt.testset1, normalize=opt.normalize, area_threshold=opt.area_threshold,
                                   ratio_threshold=opt.ratio_threshold, patch_size=(opt.patchSize, 64))
    test_set1 = torch.utils.data.Subset(test_set1, [i for i in range(len(test_set1))])
    test_data_loader1 = DataLoader(dataset=test_set1, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_set2 = PPGTrain2(root=opt.testset2, normalize=opt.normalize, area_threshold=opt.area_threshold,
                                   ratio_threshold=opt.ratio_threshold, patch_size=(opt.patchSize, 64))
    test_set2 = torch.utils.data.Subset(test_set2, [i for i in range(len(test_set2))])
    test_data_loader2 = DataLoader(dataset=test_set2, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_set3 = PPGTrain2(root=opt.testset3, normalize=opt.normalize, area_threshold=opt.area_threshold,
                                   ratio_threshold=opt.ratio_threshold, patch_size=(opt.patchSize, 64))
    test_set3 = torch.utils.data.Subset(test_set3, [i for i in range(len(test_set3))])
    test_data_loader3 = DataLoader(dataset=test_set3, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    model_description = gen_model_name(opt)
    save_path = os.path.join('clsdefect', "checkpoints", model_description)
    log_dir = 'clsdefect/records/' + model_description
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(log_dir) and not opt.resume:
        inp = input("The log dir exists, remove ([y]/n)? ")
        if inp == 'y' or inp == '':
            shutil.rmtree(log_dir)
        else:
            raise FileExistsError("Log dir exists: {}".format(log_dir))

    writer = SummaryWriter(log_dir=log_dir)

    print("===> Setting GPU")
    if opt.cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = torch.nn.DataParallel(criterion).cuda()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(itertools.chain(model.parameters(), criterion.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)

    print("===> Training")
    indices = torch.randperm(len(dataset)).tolist() if opt.shuffled else [i for i in range(len(dataset))]
    train_set = torch.utils.data.Subset(dataset, indices[:dataset.l_train])
    if dataset.l_val != 0:
        val_set = torch.utils.data.Subset(dataset, indices[-dataset.l_val:])
    else:
        val_set = None
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    if val_set:
        validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    dct = vars(opt)
    hparams = {}
    for k, v in dct.items():
        if not isinstance(v, list) and v is not None:
            hparams.update({k: v})
    writer.add_hparams(hparams, {})

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if epoch % 10 == 0:
            print("Permuting train and validation sets")
            indices = torch.randperm(len(dataset)).tolist() if opt.shuffled else [i for i in range(len(dataset))]
            train_set = torch.utils.data.Subset(dataset, indices[:dataset.l_train])
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
            if val_set:
                val_set = torch.utils.data.Subset(dataset, indices[-dataset.l_val:])
                assert len(train_set) + len(val_set) == len(dataset)
                validation_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        train(writer, training_data_loader, optimizer, criterion, epoch)
        save_checkpoint(model, criterion, epoch, opt, save_path)
        if val_set:
            val(writer, validation_data_loader, epoch)
        test(writer, test_data_loader1, epoch, idx=1)
        test(writer, test_data_loader2, epoch, idx=2)
        test(writer, test_data_loader3, epoch, idx=3)
        lr_scheduler.step()


def train(writer, data_loader, optimizer, criterion, epoch):
    global model, opt, get_classes
    model.train()
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    log = Logger(prefix="train", names=data_loader.dataset.dataset.classes, writer=writer)
    with tqdm.tqdm(total=len(data_loader), desc="Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"])) as pbar:
        for iteration, (inps, labels, masks) in enumerate(data_loader):
            if opt.cuda:
                inps = inps.cuda()
                masks = masks.cuda()
                labels = labels.cuda()

            model_output = model(inps, masks=masks, train=True)
            loss = criterion(model_output, masks, labels)
            writer.add_scalar('iter/train_loss', loss.item(), iteration + epoch*len(data_loader))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log.append(model_output['pmf'], labels, loss)
            pbar.update()

    log.write(epoch)


def val(writer, data_loader, epoch):
    global model, opt
    model.eval()
    with torch.no_grad():
        log = Logger(prefix='val', names=data_loader.dataset.dataset.classes, writer=writer, loss=False)
        with tqdm.tqdm(total=len(data_loader), desc=str('val')) as pbar:
            for iteration, (inps, labels, masks) in enumerate(data_loader):
                if opt.cuda:
                    inps = inps.cuda()
                    masks = masks.cuda()
                    labels = labels.cuda()

                model_output = model(inps, masks=masks, train=False)
                log.append(model_output['pmf'], labels)
                pbar.update()

        log.write(epoch)


def test(writer, data_loader, epoch, idx=0):
    global model, opt
    model.eval()
    with torch.no_grad():
        log = Logger(prefix=f'test{idx}', names=data_loader.dataset.dataset.classes, writer=writer, loss=False,
                            model=model)
        with tqdm.tqdm(total=len(data_loader), desc=str('test')) as pbar:
            for iteration, (inps, labels, masks) in enumerate(data_loader):
                if opt.cuda:
                    inps = inps.cuda()
                    masks = masks.cuda()
                    labels = labels.cuda()

                model_output = model(inps, masks=masks, train=False)
                log.append(model_output['pmf'], labels)
                pbar.update()

        log.write(epoch, th=opt.thresholds)


if __name__ == "__main__":
    main()
