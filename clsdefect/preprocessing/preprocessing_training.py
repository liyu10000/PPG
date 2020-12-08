"""
* This file is used for preprocessing of training data
* It is expected that inputs are folders each has an image file and a collection of csv files representing different
  defected regions.
* Output is a folder which contains patches and a pkl file that has information about different defects in that patch.
"""
import pandas as pd
import warnings
import os
from PIL import ImageDraw
from PIL import Image
from PIL import ImageChops
from torchvision.transforms import functional as F
import tqdm
import torch
import torchvision
from ..utils import list_all_files_sorted, list_all_folders, load_obj, save_obj

classes = ['corrosion', 'fouling', 'delamination']


def generate_binary_mask(image: Image, coords: list, defect_type=0):
    """
    generates a binary mask that is =1 inside the given coordinates and =0 outside
    :param image: The image to use for that
    :param coords: The coordinates of the polygon to fill given in the form of list of tuples [(xi,yi) for xi,yi in points]
    :param defect_type: The defect  type used in coloring the polygon uniquely according to the defect type
    :return: two images, the first is the binary mask and the second is the original image where the polygon is highligthed
    """
    W, H = image.size
    mask = Image.new("1", (W, H))
    draw_standalone = ImageDraw.Draw(mask)
    draw_standalone.polygon(coords, fill=True)
    draw_over = ImageDraw.Draw(image.convert("RGB"))
    fill = [0 for _ in classes]
    fill[defect_type] = 128
    draw_over.polygon(coords, fill=tuple(fill))
    return mask, image


def intersect(mask1: Image, mask2: Image):
    """
    Calculates the binary mask of the intersection of both input masks
    :param mask1:
    :param mask2:
    :return: binary intersection map
    """
    return ImageChops.logical_and(mask1, mask2)


def union(mask1: Image, mask2: Image):
    """
    Calculates the binary mask of the union of both input masks
    :param mask1:
    :param mask2:
    :return: binary union map
    """
    return ImageChops.logical_or(mask1, mask2)


def get_coords(file: str):
    coords = pd.read_csv(file, sep=',')
    coords = [(x, y) for x, y in zip(coords['X'], coords['Y'])]
    return coords


def get_label(file: str):
    label = -1
    for i, cls in enumerate(classes):
        if cls in file:
            if label == -1:
                label = i
            else:
                Warning("Another label has already been defined!")
                Warning("Label: {}, File: {}".format(label, file))
            # break
    assert label != -1
    return label


def get_label_name(label: int):
    return classes[label]


def calculate_area(mask: Image):
    """Return the number of pixels in mask that are True and the ratio from all image as well
    img must be a PIL.Image object in mode "1".
    """
    bbox = mask.getbbox()
    W, H = mask.size
    if not bbox:
        return 0, 0
    active = sum(mask.crop(bbox).point(lambda x: 255 if x else 0).convert("L").point(bool).getdata())
    # active = sum(mask.crop(bbox).getdata())
    return active, active/(W*H)


resize = F.resize


def get_bbox(coords: list):
    xmin = min([i for (i, j) in coords])
    xmax = max([i for (i, j) in coords])
    ymin = min([j for (i, j) in coords])
    ymax = max([j for (i, j) in coords])
    return [xmin, ymin, xmax, ymax]


def get_box_coords(box: list):
    return [(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1]), (box[0], box[1])]


def find_overlapping(image: Image, box: list, threshold: float, mask: Image):
    box_coords = get_box_coords(box)
    rect, _ = generate_binary_mask(image, box_coords)
    area_rect, _ = calculate_area(rect)
    overlapping = []
    masks = [d for d in mask.split()]
    defects = [m.convert('1') for m in masks]
    combined_mask = []
    for label, defect, mask_ in zip(classes, defects, masks):
        mask_cropped = intersect(rect, defect)
        intersecting_area, _ = calculate_area(mask_cropped)
        ratio = intersecting_area / area_rect
        assert 0 <= ratio <= 1
        if ratio >= threshold:
            overlapping.append({'ratio': ratio, 'area': intersecting_area, 'label': label, 'mask': mask_cropped, 'mask_cropped': mask_cropped.crop(box)})
        combined_mask.append(mask_.crop(box))
    if combined_mask:
        combined_mask = Image.merge('RGB', combined_mask)
    return overlapping, combined_mask


def calculate_number_patches(W, H, patch_size, stride_size):
    nW = (W - patch_size) // stride_size + 1
    nH = (H - patch_size) // stride_size + 1
    return nW, nH


def extract_patch(image: Image, mask: Image, iW, iH, patch_size, stride_size):
    box = [iW*stride_size, iH*stride_size, iW*stride_size + patch_size, iH*stride_size + patch_size]
    return image.crop(box), mask.crop(box), box


def generate_3layer_mask(folder: str, image: Image):
    mask = []
    W, H = image.size
    files = list_all_files_sorted(folder, post=".csv")
    files = [[file for file in files if cls in file] for cls in classes]
    for i, cls in enumerate(classes):
        if len(files[i]) == 0:
            mask.append(torch.zeros((1, H, W)))
            continue

        try:
            coords = get_coords(files[i][0])
            temp_mask = generate_binary_mask(image, coords, i)[0]
        except:
            print("this file has issues")
            print(files[i][0])

        if len(files[i]) > 1:
            for d in files[i][1:]:
                try:
                    coords = get_coords(d)
                except:
                    print("this file has issues")
                    print(d)
                    continue
                temp_mask = union(temp_mask, generate_binary_mask(image, coords, i)[0])
        mask.append(torchvision.transforms.functional.to_tensor(temp_mask))
    mask = torch.cat(mask, dim=0).to(dtype=torch.float)
    mask = torchvision.transforms.functional.to_pil_image(mask)
    return mask


def write_patches(src_root: str, dst_root: str, patch_size: int, stride_size: int, threshold: float):
    os.makedirs(dst_root, exist_ok=True)
    images_phrases = ["*HR.png", "*HR.jpg", "*SR.png", "*SR.jpg"]  # a keyword to get the image file
    images_folders = list_all_folders(src_root, isFull=True)
    img_counter = []
    try:
        df = load_obj(os.path.join(dst_root, 'sofar.pkl'))
        imx_start = df['imx']
        img_counter.extend(df['img_counter'])
        print(">>>> RESUMING <<<<")
    except:
        imx_start = -1

    with tqdm.tqdm(total=len(images_folders), desc="") as bar_out:
        for imx, folder in enumerate(images_folders):
            if imx <= imx_start: # and imx != 3:
                continue
            name = folder.split('/')[-1]
            counter = 0
            img_counter.append(0)
            if len(list_all_folders(folder)) == 1:
                folder = os.path.join(folder, list_all_folders(folder)[0])

            flag = False
            for idx, phrase in enumerate(images_phrases):
                if idx > 0:
                    warnings.warn("checking other phrases {}".format(phrase))
                if len(list_all_files_sorted(folder, phrase=phrase)) == 1:
                    flag = True
                    image_file = list_all_files_sorted(folder, phrase=phrase)[0]
                    break
            if not flag or not image_file:
                raise FileExistsError(f"that image doesn't exist (skipping): {folder}")
                continue

            image = Image.open(image_file).convert("RGB")
            W, H = image.size
            nW, nH = calculate_number_patches(W, H, patch_size, stride_size)
            mask = generate_3layer_mask(folder, image)
            with tqdm.tqdm(total=nW*nH, desc=f"image {folder.split('/')[-1]}-{dst_root}") as pbar:
                for iw in range(nW):
                    for ih in range(nH):
                        patch, _, box = extract_patch(image, mask, iw, ih, patch_size, stride_size)
                        overlapping, combined_mask = find_overlapping(image, box, threshold, mask)
                        if len(overlapping) != 0:
                            patch.save(os.path.join(dst_root, f"{name}_{counter}_patch.png"), "PNG")
                            combined_mask.save(os.path.join(dst_root, f"{name}_{counter}_mask.png"), "PNG")
                            save_obj({'overlapping': overlapping,
                                            'image': folder,
                                            'iw, ih': (iw, ih),
                                            'patch_size': patch_size,
                                            'stride_size': stride_size}, os.path.join(dst_root, f'{name}_{counter}_meta.pkl'))
                            counter += 1
                            img_counter[-1] += 1
                    pbar.update(nH)

            data_frame = {'images': images_folders, 'imx': imx, 'folder': folder, 'counter': counter, 'img_counter': img_counter}
            save_obj(data_frame, os.path.join(dst_root, "sofar.pkl"))
            bar_out.set_description(f"Done with {img_counter[-1]}/{sum(img_counter)} images")
            bar_out.update()

    print("Done all with images counter {}".format(img_counter))


if __name__ == "__main__":
    patch_size = 64
    folder = 'batch10'
    th = 1e-03
    write_patches(src_root=f'/home/krm/ext/PPG/Classification_datasets/PPG/{folder}/',
                  dst_root=f'/home/krm/ext/PPG/Classification_datasets/PPG/{folder}_{patch_size}_{th}/',
                  patch_size=patch_size, stride_size=int(patch_size//2), threshold=th)
