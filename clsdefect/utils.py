import argparse
import torch
import torchvision.transforms as T
import glob
import os
import pandas as pd
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import numpy as np
import random
import sys
import matplotlib.pyplot
import cv2
import seaborn as sn
import pickle
from sklearn import metrics
from typing import Union


def collate_fn(batch):
    return tuple(zip(*batch))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def list_all_folders(folderName, isFull=False):
    if isFull:
        return sorted([os.path.join(folderName, folder) for folder in next(os.walk(folderName))[1]])
    else:
        return sorted(next(os.walk(folderName))[1])


def list_all_files_sorted(folderName, pre="", post="", phrase=None):
    assert phrase is None or (pre == "" and post == "")
    if phrase is None:
        return sorted(glob.glob(os.path.join(folderName, pre + "*" + post)))
    else:
        return sorted(glob.glob(os.path.join(folderName, phrase)))


def convert_csv_images_list(src_path):
    """
    :param src_path: the source path where csv files are located
    :return: image, draw_over, standalone mask
    """
    images = list_all_folders(src_path)
    out_lists = [], [], []
    for image_folder in images:
        outs = convert_csv_image(os.path.join(src_path, image_folder))
        [out_list.append(out) for out, out_list in zip(outs, out_lists)]
    return out_lists  # images list, drawing over list, drawing standalone list


def convert_csv_image(src_path, style: Union['combined', 'individual']):
    """
    :param src_path: the source path where csv files are located
    :return: image, drawing over, drawing standalone
    """
    to_tensor = T.ToTensor()
    image_file = list_all_files_sorted(src_path, post="*.png")
    if len(image_file) == 2:
        image_file = [im for im in image_file if 'defects' not in im.lower()]
    assert len(image_file) == 1
    image_file = image_file[0]
    image_orig = Image.open(image_file).convert('RGB')
    image_orig = to_tensor(image_orig)
    image = image_orig.clone()
    complete_mask = torch.zeros_like(image)
    H, W = image.shape[1:]
    defects_csv = list_all_files_sorted(src_path, post=".csv")
    corrosion_index, fouling_index, delamination_index = 0, 0, 0
    for file in defects_csv:
        coords = pd.read_csv(file, sep=',')
        coords = [(x, y) for x, y in zip(coords['X'], coords['Y'])]
        assert ('corrosion' in file.lower()) or ('fouling' in file.lower()) or ('delamination' in file.lower())
        fig = Image.new("RGB", (W, H))
        draw_standalone = ImageDraw.Draw(fig)
        if 'corrosion' in file.lower():
            corrosion_index += 1
            if style == 'individual':
                draw_standalone.polygon(coords, fill=(corrosion_index, 0, 0))
            else:
                draw_standalone.polygon(coords, fill=(255, 0, 0))
            tp, ind = 'corrosion', corrosion_index
        elif 'delamination' in file.lower():
            delamination_index += 1
            if style == 'individual':
                draw_standalone.polygon(coords, fill=(0, 0, delamination_index))
            else:
                draw_standalone.polygon(coords, fill=(0, 0, 255))
            tp, ind = 'delamination', delamination_index
        else:
            fouling_index += 1
            if style == 'individual':
                draw_standalone.polygon(coords, fill=(0, fouling_index, 0))
            else:
                draw_standalone.polygon(coords, fill=(0, 255, 0))
            tp, ind = 'fouling', fouling_index
        mask = to_tensor(fig)
        image += mask
        complete_mask += mask
    image[image > 1] = 1
    image[image < 0] = 0
    complete_mask[complete_mask > 1] = 1
    complete_mask[complete_mask < 0] = 0
    return image_orig, image, complete_mask


def add_confusion_matrix(writer, epoch, confusion_matrix, labels, name='train/confusion_matrix', normalize=False):
    matplotlib.use('Agg')
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') * 10 / confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = np.nan_to_num(confusion_matrix, copy=True)
        confusion_matrix = confusion_matrix.astype('int')

    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    np.set_printoptions(precision=3)
    fig = matplotlib.pyplot.figure(figsize=(5, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    sn.set(font_scale=1.4)  # for label size originally 1.4
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 18}, ax=ax)
    ax.set_xlabel('Prediction', fontsize=14)  # originally 7
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('True Label', fontsize=14)  # originally 7
    ax.yaxis.set_label_position('left')
    fig.set_tight_layout(True)
    writer.add_figure(name, fig, epoch)
    return fig


def add_roc(writer, epoch, true, pmf, name='train/ROC'):
    matplotlib.use('Agg')
    fpr, tpr, thresholds = metrics.roc_curve(true, pmf)
    roc_auc = metrics.auc(fpr, tpr)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    fig = matplotlib.pyplot.figure(figsize=(5, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    lw = 2
    ax.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best th={thresholds[ix]:.5f}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name}')
    ax.legend(loc="lower right")
    fig.set_tight_layout(True)
    if writer is not None:
        writer.add_figure(name, fig, epoch)
    return fpr, tpr, roc_auc, thresholds[ix], fig


def add_prec_rec_curve(writer, epoch, true, pmf, name='train/Precision vs Recall'):
    matplotlib.use('Agg')
    precision, recall, thresholds = metrics.precision_recall_curve(true, pmf)
    fig = matplotlib.pyplot.figure(figsize=(5, 4), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    lw = 2
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall + 1e-6)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    ax.plot(recall, precision, color='darkorange', marker=".", lw=lw, label="PRC")
    no_skill = sum([1 for t in true if t == 1]) / len(true)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.scatter(recall[ix], precision[ix], marker='o', color='black', label=f'Best th={thresholds[ix]:.5f}')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{name}')
    ax.legend(loc="lower left")
    fig.set_tight_layout(True)
    writer.add_figure(name, fig, epoch)
    return precision, recall, thresholds[ix], fig


def add_plot(writer, epoch, xvalues, yvalues, name='train/plot', type='bar'):
    matplotlib.use('Agg')
    assert type == 'bar' or type == 'plot'
    np.set_printoptions(precision=2)
    fig = matplotlib.pyplot.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    if type == 'bar':
        ax.bar(xvalues, yvalues)
    elif type == 'plot':
        ax.plot(xvalues, yvalues)
    fig.set_tight_layout(True)
    writer.add_figure(name, fig, epoch)
    return fig


def save_checkpoint(model, criterion, epoch, opt, save_path, extra: dict = None):
    model_out_path = os.path.join(save_path, "{}.pth".format(epoch))
    if opt.cuda:
        model_file = model.module.__class__.__module__
        model_class = model.module.__class__.__name__
        model_state = model.module.state_dict()
        loss_file = criterion.module.__class__.__module__
        loss_class = criterion.module.__class__.__name__
        loss_state = criterion.module.state_dict()
    else:
        model_file = model.__class__.__module__
        model_class = model.__class__.__name__
        model_state = model.state_dict()
        loss_file = criterion.__class__.__module__
        loss_class = criterion.__class__.__name__
        loss_state = criterion.state_dict()

    state = {"epoch": epoch,
             "opt": opt,
             "args": sys.argv,
             "model_file": model_file,
             "model_class": model_class,
             "model_state": model_state,
             "loss_file": loss_file,
             "loss_class": loss_class,
             "loss_state": loss_state,
             }

    if extra: state.update(extra.items())

    # check path status
    os.makedirs(save_path, exist_ok=True)

    # save model
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def load_checkpoint(model, criterion, path, **extra_params):
    if 'iscuda' in extra_params:
        iscuda = extra_params.pop('iscuda')
    else:
        iscuda = torch.cuda.is_available()
    if iscuda:
        dict = torch.load(path, map_location='cuda')
    else:
        dict = torch.load(path, map_location='cpu')
    print(dict["model_file"])
    print(dict["model_class"])
    model.load_state_dict(dict["model_state"])
    print(dict["loss_file"])
    print(dict["loss_class"])
    if criterion: criterion.load_state_dict(dict["loss_state"])
    return model, criterion, dict["epoch"], dict


def set_seed(seed, cuda, benchmark=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # needed
        if benchmark: torch.backends.cudnn.benchmark = False
    return


def tensor_show_write(tensor, name='tensor', wait_value=2000, folder="", show=True, write=False, isCV=True):
    def convert_cv(tensor):
        if isCV:
            return tensor
        else:
            return tensor[..., ::-1]

    if write:
        os.makedirs('./results', exist_ok=True)
    if len(tensor.shape) == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if folder != "" and write:
        write_path = os.path.join('./results', folder)
        os.makedirs(write_path, exist_ok=True)
    elif write:
        write_path = './results'
        os.makedirs(write_path, exist_ok=True)
    else:
        write_path = False
    if len(tensor.shape) == 4:
        t = (tensor.transpose(1, 2).transpose(2, 3).data.numpy() * 255.0).astype(np.uint8)
        for i in range(t.shape[0]):
            if show:
                cv2.imshow("{}/{}_{}".format(folder, name, i), convert_cv(t[i, :, :, :]))
                cv2.waitKey(wait_value)
            if write:
                assert write_path
                cv2.imwrite("{}/{}_{}.png".format(write_path, name, i), convert_cv(t[i, :, :, :]))
    elif len(tensor.shape) == 3:
        t = (tensor.transpose(0, 1).transpose(1, 2).data.numpy() * 255.0).astype(np.uint8)
        if show:
            cv2.imshow("{}/{}".format(folder, name), convert_cv(t))
            cv2.waitKey(wait_value)
        if write:
            assert write_path
            cv2.imwrite("{}/{}.png".format(write_path, name), convert_cv(t))
    elif len(tensor.shape) == 2:
        t = (tensor.data.numpy() * 255.0).astype(np.uint8)
        if show:
            cv2.imshow("{}/{}".format(folder, name), t)
            cv2.waitKey(wait_value)
        if write:
            cv2.imwrite("{}/{}.png".format(folder, name), t)
    else:
        raise SyntaxError('This shape of tensor is not supported, tensor.shape=' + str(tensor.shape))


def get_transform(train, resize=None):
    transforms = []
    transforms.append(transforms.ToTensor())
    if train:
        transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms)


def gen_model_name(opts,
                   omit_keys=['num_classes', 'gpus', 'resume', 'start_epoch', 'threads', 'cuda', 'seed', 'nEpochs',
                              'batchSize', 'testset_HR', 'testset_SR']):
    name = opts.model
    if opts.ID != "":
        name += '_' + str(opts.ID)
    if not opts.pretrained:
        name += '_untrained'

    if hasattr(opts, "dataset"):
        if opts.dataset:
            name += '_' + os.path.basename(opts.dataset)

    if hasattr(opts, "testset"):
        if opts.testset:
            name += '_' + os.path.basename(opts.testset)
    # if not opts.normalize:
    #     name += "_unnormalized"
    # if opts.crop != 0:
    #     name += "_crop:{}".format(opts.crop)
    omit_keys.extend(['model', 'ID', 'pretrained', 'dataset', 'testset', 'normalize', 'crop', 'patchSize', 'test_loss',
                      'test_loss_params', 'datasets', 'testset1', 'testset2', 'testset3', 'testset4', 'testset5',
                      'testset6', 'test_csv_files', 'weight_decay'])
    for key, value in opts.__dict__.items():
        if key not in omit_keys and value != "":
            name += '_' + str(key) + ':' + str(value)
    name = name.replace('/', ':')
    print("===>Model Description: {}".format(name))
    return name


def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
