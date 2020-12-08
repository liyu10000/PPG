import argparse
import importlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision

from clsdefect.Datasets import PPGTestShipPartSegmList, PPGImgShipPartPatches
from clsdefect.utils import list_all_files_sorted, save_obj, load_checkpoint
from config import cfg

# matplotlib.use('Agg')
plt.ioff()


parser = argparse.ArgumentParser(description="PPG defects classification test code options")
parser.add_argument("--seg_out_path", type=str, help="folder location of the output of defect segmentation",
                    default=cfg.segdefect.orisize_pred_joint_dir)
parser.add_argument("--ship_out_path", type=str, help="folder location of the output of whole ship segmentation",
                    default=cfg.segwhole.orisize_pred_dir)
parser.add_argument("--part_out_path", type=str, help="folder location of the output of part segmentation",
                    default=cfg.segpart.orisize_pred_dir)
parser.add_argument("--inp_path", type=str, help="folder location of the input images",
                    default=cfg.test_image_dir)
parser.add_argument("--out_path", type=str, help="folder location of the input images",
                    default=cfg.clsdefect.out_path)
parser.add_argument("--model_path", type=str, help="file path to the model",
                    default=cfg.clsdefect.model_path)
parser.add_argument("--batchSize", type=int, help="batch size to work on simultaneously",
                    default=cfg.clsdefect.test_batch_size)
parser.add_argument("--patchSize", type=int, help="patch size for defect labeling process",
                    default=cfg.clsdefect.test_patch_size)
parser.add_argument("--strideSize", type=int, help="stride size for defect labeling process",
                    default=cfg.clsdefect.test_stride_size)
parser.add_argument("--ratio_threshold", type=float, default=0.1,
                    help="area percentage to consider wether to label a patch is defected or not")
parser.add_argument("--gpu", type=str, default=str(cfg.gpu), help="Use GPUs? takes the ID of the GPU")
parser.add_argument("--thresholds", type=list, default=[0.5, 0.5, 0.24],
                    help="thresholds used to classify as defected (order: corrosion, fouling, delamination)")


def refine3(classes, mask):
    mask = sum([mask[i] for i in range(3)]) > 0
    return classes*mask


def thresholding(mask, thresholds=[0.35, 0.35, 0.05]):
    new_mask = torch.clone(mask)
    for i, th in enumerate(thresholds):
        # mask[i] = map(lambda x: float(x >= th), mask[i])
        new_mask[i] = mask[i].ge(th).float()
    return new_mask


def normalizing(mask, thresholds=[0.35, 0.35, 0.05]):
    new_mask = torch.clone(mask)
    for i, th in enumerate(thresholds):
        # mask[i] = map(lambda x: float(x >= th), mask[i])
        isgreater = mask[i].ge(th).float()
        new_mask[i] = 0.5 + (mask[i] - th) / (2*(isgreater*(1 - th) + (1 - isgreater)*(th)))
    return new_mask


def combine_imgs(original: Image, images: list, titles: list or None = None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(4, 1.47))
    cnt = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(2, n_images, cnt)
        plt.imshow(image)
        a.set_title(title)
        a.axis('off')
        a.margins(0, 0)
        cnt += 1

    alpha = 0.8
    for image, title in zip(images, titles):
        # print(title)
        a = fig.add_subplot(2, n_images, cnt)
        plt.imshow(Image.blend(image, original, alpha))
        a.axis('off')
        a.margins(0, 0)
        cnt += 1

    legened_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=15, label='corrosion'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=15, label='fouling'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=15, label='delamination'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='c', markersize=15, label='foul. + delam.'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='m', markersize=15, label='corr. + delam.'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=15, label='corr. + foul.'),
        plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='w', markersize=15, label='corr. + foul. + delam.'),
    ]
    fig.legend(frameon=False, loc='lower center', handles=legened_elements, ncol=7)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    return fig


def main():
    opt = parser.parse_args()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    seg_out_path = opt.seg_out_path
    ship_out_path = opt.ship_out_path
    part_out_path = opt.part_out_path
    inp_path = opt.inp_path
    out_path = opt.out_path
    ckpt = opt.model_path
    patch_size = opt.patchSize
    stride_size = opt.strideSize
    batch_size = opt.batchSize
    thresholds = opt.thresholds
    ratio_threshold = opt.ratio_threshold

    eps = 1e-6  # for numerical stability
    model_name = "different_warper"
    model_params = f"64_64_0.5"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    assert patch_size % stride_size == 0
    mean_scale = (patch_size // stride_size)**2

    print('======> loading dataset')
    data = PPGTestShipPartSegmList(roots=[inp_path], names='new', ship_results_path=ship_out_path,
                                    part_results_path=part_out_path, seg_results_path=seg_out_path,
                                    normalize=True)

    print('======> loading model')
    mod = importlib.import_module(f"clsdefect.models.{model_name}")
    model = mod.Model(num_classes=3, params=model_params, pretrained=True)
    model, _, _, _ = load_checkpoint(model, None, ckpt, num_classes=3, model_params="")
    model = model.to(device)
    model.eval()

    comp_path = os.path.join(out_path, 'comparisons')
    percentages_path = os.path.join(out_path, 'percentages')
    save_path = out_path
    os.makedirs(comp_path, exist_ok=True)
    os.makedirs(percentages_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for outer_idx, batch in enumerate(tqdm(data, desc="images")):
            inp, ship_mask, part_mask, seg_mask, cls_mask, img, name = batch
            patch_data = PPGImgShipPartPatches(inp, ship_mask, part_mask, seg_mask, cls_mask, img, patch_size=patch_size, stride_size=stride_size, ratio_threshold=ratio_threshold)
            loader = DataLoader(patch_data, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            nH, nW = (patch_data.nH - 1) * stride_size + patch_size, (patch_data.nW - 1) * stride_size + patch_size
            image = torch.zeros([3, nH, nW], dtype=torch.float, device=device)  # patch_data.img
            image_normalized = torch.zeros([3, nH, nW], dtype=torch.float, device=device)
            pmf = torch.zeros([3, nH, nW], dtype=torch.float, device=device)
            prediction = torch.zeros([3, nH, nW], dtype=torch.float, device=device)
            true = torch.zeros([3, nH, nW], dtype=torch.float, device=device)
            ship_out = torch.zeros([1, nH, nW], dtype=torch.float, device=device)
            seg_out = torch.zeros([1, nH, nW], dtype=torch.float, device=device)
            part_out = torch.zeros([3, nH, nW], dtype=torch.float, device=device)
            for ib, batch in enumerate(tqdm(loader, desc="patches")):
                batch = [e.to(device) for e in batch]
                patches, ship_patches, part_patches, seg_patches, cls_patches, img_patches, labels, coords = batch
                out = model(patches.to(device))
                pmf_out = out['pmf']
                for c, p, shm, pm, sm, cm, norm_im, im in \
                        zip(coords.split(1, dim=0), pmf_out.split(1, dim=0), ship_patches.split(1, dim=0),
                            part_patches.split(1, dim=0), seg_patches.split(1, dim=0), cls_patches.split(1, dim=0),
                            patches.split(1, dim=0), img_patches.split(1, dim=0)):
                    c = c.squeeze()
                    p = p.squeeze().unsqueeze(dim=1).unsqueeze(dim=1)
                    shm = shm.squeeze()
                    pm = pm.squeeze()
                    sm = sm.squeeze()
                    cm = cm.squeeze()
                    im = im.squeeze()

                    sl_h = slice(c[0] * stride_size, c[0] * stride_size + patch_size)
                    sl_w = slice(c[1] * stride_size, c[1] * stride_size + patch_size)
                    norm_im = norm_im.squeeze()
                    pmf[:, sl_h, sl_w] += p
                    prediction[:, sl_h, sl_w] += thresholding(p, thresholds)
                    true[:, sl_h, sl_w] += cm
                    ship_out[:, sl_h, sl_w] += shm
                    seg_out[:, sl_h, sl_w] += sm
                    part_out[:, sl_h, sl_w] += pm
                    image[:, sl_h, sl_w] += im
                    image_normalized[:, sl_h, sl_w] += norm_im

            pmf = pmf / mean_scale
            prediction = prediction / mean_scale
            true = true / mean_scale
            image = image / mean_scale
            image_normalized = image_normalized / mean_scale
            part_out /= mean_scale
            seg_out /= mean_scale
            ship_out /= mean_scale

            classes_thresholded = (thresholding(pmf, thresholds)*seg_out*ship_out)
            cor, foul, delam = classes_thresholded.split(1, dim=0)
            VS, BT, TS = part_out.split(1, 0)
            foul = foul * (1 - TS)
            classes_refined = torch.cat((cor, foul, delam), 0)
            image_classes = torchvision.transforms.functional.to_pil_image(classes_thresholded.cpu(), 'RGB')
            image_classes_refined = torchvision.transforms.functional.to_pil_image(classes_refined.cpu(), 'RGB')
            images = [
                torchvision.transforms.functional.to_pil_image(ship_out.cpu().repeat(3, 1, 1), 'RGB'),
                torchvision.transforms.functional.to_pil_image(part_out.cpu(), 'RGB'),
                torchvision.transforms.functional.to_pil_image(seg_out.cpu().repeat(3, 1, 1), 'RGB'),
                image_classes,
                image_classes_refined,
            ]
            titles = [
                'Whole Ship Segm',
                'Part Segm Output',
                'Defect Segm Output',
                'Prediction (thresholded)',
                'Predictions (employing Part Segm Output)',
            ]
            fig = combine_imgs(original=torchvision.transforms.functional.to_pil_image(image.cpu(), 'RGB'),
                               images=images, titles=titles)
            plt.gca().set_axis_off()
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(os.path.join(comp_path, f'{os.path.splitext(os.path.basename(name))[0]}.png'), bbox_inches='tight', pad_inches=0)

            for im, title in zip(images[-2:], titles[-2:]):
                path = os.path.join(save_path, title)
                os.makedirs(path, exist_ok=True)
                im.save(os.path.join(path, f'{os.path.splitext(os.path.basename(name))[0]}.png'))

            whole_ship_pixel_count = ship_out.sum().item()

            dct = {}
            dct["corrosion_before_refinement_pixel_count"] = classes_thresholded[0].sum().item()
            dct["corrosion_after_refinement_pixel_count"] = classes_refined[0].sum().item()
            dct["fouling_before_refinement_pixel_count"] = classes_thresholded[1].sum().item()
            dct["fouling_after_refinement_pixel_count"] = classes_refined[1].sum().item()
            dct["delamination_before_refinement_pixel_count"] = classes_thresholded[2].sum().item()
            dct["delamination_after_refinement_pixel_count"] = classes_refined[2].sum().item()
            TS_pixel_count = TS.sum().item()
            BT_pixel_count = BT.sum().item()
            VS_pixel_count = VS.sum().item()
            #
            # for name_dict, name_cls in [('before_refinement', 'classes_thresholded'), ('after_refinement', 'classes_refined')]:
            #     for idt, def_type in enumerate(['corrosion', 'fouling', 'delamination']):
            #         exec(f'{def_type}_{name_dict}_pixel_count = {name_cls}[{idt}].sum().item()')

            csv_out = {'ship_area': ship_mask.sum().item(),
                       'total': {
                           'before_refinement': {
                               'pixel_count': {'corrosion': classes_thresholded[0].sum().item(),
                                               'fouling': classes_thresholded[1].sum().item(),
                                               'delamination': classes_thresholded[2].sum().item()},
                               'area_ratio': {'corrosion': classes_thresholded[0].sum().item() / (whole_ship_pixel_count + eps),
                                              'fouling': classes_thresholded[1].sum().item() / (whole_ship_pixel_count + eps),
                                              'delamination': classes_thresholded[2].sum().item() / (whole_ship_pixel_count + eps)},
                           },
                           'after_refinement': {
                               'pixel_count': {'corrosion': classes_refined[0].sum().item(),
                                               'fouling': classes_refined[1].sum().item(),
                                               'delamination': classes_refined[2].sum().item()},
                               'area_ratio': {'corrosion': classes_refined[0].sum().item() / (whole_ship_pixel_count + eps),
                                              'fouling': classes_refined[1].sum().item() / (whole_ship_pixel_count + eps),
                                              'delamination': classes_refined[2].sum().item() / (whole_ship_pixel_count + eps)},
                           },
                       },
                       'TS': {
                           'before_refinement': {
                               'pixel_count': {'corrosion': (classes_thresholded[0]*TS).sum().item(),
                                               'fouling': (classes_thresholded[1]*TS).sum().item(),
                                               'delamination': (classes_thresholded[2]*TS).sum().item()},
                               'area_ratio': {
                                   'corrosion': (classes_thresholded[0]*TS).sum().item() / (TS_pixel_count + eps),
                                   'fouling': (classes_thresholded[1]*TS).sum().item() / (TS_pixel_count + eps),
                                   'delamination': (classes_thresholded[2]*TS).sum().item() / (TS_pixel_count + eps)},
                           },
                           'after_refinement': {
                               'pixel_count': {'corrosion': (classes_refined[0]*TS).sum().item(),
                                               'fouling': (classes_refined[1]*TS).sum().item(),
                                               'delamination': (classes_refined[2]*TS).sum().item()},
                               'area_ratio': {
                                   'corrosion': (classes_refined[0]*TS).sum().item() / (TS_pixel_count + eps),
                                   'fouling': (classes_refined[1]*TS).sum().item() / (TS_pixel_count + eps),
                                   'delamination': (classes_refined[2]*TS).sum().item() / (TS_pixel_count + eps)},
                           },
                       },
                       'BT': {
                           'before_refinement': {
                               'pixel_count': {'corrosion': (classes_thresholded[0] * BT).sum().item(),
                                               'fouling': (classes_thresholded[1] * BT).sum().item(),
                                               'delamination': (classes_thresholded[2] * BT).sum().item()},
                               'area_ratio': {
                                   'corrosion': (classes_thresholded[0] * BT).sum().item() / (BT_pixel_count + eps),
                                   'fouling': (classes_thresholded[1] * BT).sum().item() / (BT_pixel_count + eps),
                                   'delamination': (classes_thresholded[2] * BT).sum().item() / (BT_pixel_count + eps)},
                           },
                           'after_refinement': {
                               'pixel_count': {'corrosion': (classes_refined[0] * BT).sum().item(),
                                               'fouling': (classes_refined[1] * BT).sum().item(),
                                               'delamination': (classes_refined[2] * BT).sum().item()},
                               'area_ratio': {
                                   'corrosion': (classes_refined[0] * BT).sum().item() / (BT_pixel_count + eps),
                                   'fouling': (classes_refined[1] * BT).sum().item() / (BT_pixel_count + eps),
                                   'delamination': (classes_refined[2] * BT).sum().item() / (BT_pixel_count + eps)},
                           },
                       },
                       'VS': {
                           'before_refinement': {
                               'pixel_count': {'corrosion': (classes_thresholded[0] * VS).sum().item(),
                                               'fouling': (classes_thresholded[1] * VS).sum().item(),
                                               'delamination': (classes_thresholded[2] * VS).sum().item()},
                               'area_ratio': {
                                   'corrosion': (classes_thresholded[0] * VS).sum().item() / (VS_pixel_count + eps),
                                   'fouling': (classes_thresholded[1] * VS).sum().item() / (VS_pixel_count + eps),
                                   'delamination': (classes_thresholded[2] * VS).sum().item() / (VS_pixel_count + eps)},
                           },
                           'after_refinement': {
                               'pixel_count': {'corrosion': (classes_refined[0] * VS).sum().item(),
                                               'fouling': (classes_refined[1] * VS).sum().item(),
                                               'delamination': (classes_refined[2] * VS).sum().item()},
                               'area_ratio': {
                                   'corrosion': (classes_refined[0] * VS).sum().item() / (VS_pixel_count + eps),
                                   'fouling': (classes_refined[1] * VS).sum().item() / (VS_pixel_count + eps),
                                   'delamination': (classes_refined[2] * VS).sum().item() / (VS_pixel_count + eps)},
                           },
                       },
                       }

            save_obj(csv_out, os.path.join(percentages_path, f'{os.path.splitext(os.path.basename(name))[0]}.pkl'))
            pd.DataFrame.from_dict({'areas': csv_out}).to_csv(os.path.join(percentages_path, f'{os.path.splitext(os.path.basename(name))[0]}.txt'))


if __name__ == "__main__":
    main()
