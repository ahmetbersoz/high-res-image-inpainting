from PIL import Image
import copy
import sys, getopt
import os
import cv2
import torch
import numpy as np
from torch import nn
from skimage import io, transform
import scipy.ndimage
from DFNet_core import DFNet
import matplotlib.pyplot as plt
from RefinementNet_core import RefinementNet

device = torch.device('cpu')

def to_numpy(tensor):
    tensor = tensor.mul(255).byte().data.cpu().numpy()
    tensor = np.transpose(tensor, [0, 2, 3, 1])
    return tensor

def padding(img, height=512, width=512, channels=3):
    channels = img.shape[2] if len(img.shape) > 2 else 1
    interpolation=cv2.INTER_NEAREST
    
    if channels == 1:
        img_padded = np.zeros((height, width), dtype=img.dtype)
    else:
        img_padded = np.zeros((height, width, channels), dtype=img.dtype)

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width
    new_cols = width
    new_rows = height
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = cv2.resize(img, (new_cols, height), interpolation=interpolation)
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        img = cv2.resize(img, (width, new_rows), interpolation=interpolation)
        if new_rows > height:
            new_rows = height
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img
    return img_padded, new_cols, new_rows

def preprocess_image_dfnet(image, mask, model):
    image, new_cols, new_rows = padding(image, 512, 512)
    mask, _, _ = padding(mask, 512, 512)
    image = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.uint8)
    mask = np.ascontiguousarray(np.expand_dims(mask, 0)).astype(np.uint8)

    image = torch.from_numpy(image).to(device).float().div(255)
    mask = 1 - torch.from_numpy(mask).to(device).float().div(255)
    image_miss = image * mask
    DFNET_output = model(image_miss.unsqueeze(0), mask.unsqueeze(0))[0]
    DFNET_output = image * mask + DFNET_output * (1 - mask)
    DFNET_output = to_numpy(DFNET_output)[0]
    DFNET_output = cv2.cvtColor(DFNET_output, cv2.COLOR_BGR2RGB)
    DFNET_output = DFNET_output[(DFNET_output.shape[0] - new_rows) // 2: (DFNET_output.shape[0] - new_rows) // 2 + new_rows, 
            (DFNET_output.shape[1] - new_cols) // 2: (DFNET_output.shape[1] - new_cols) // 2 + new_cols, ...]

    return DFNET_output

def preprocess_image(image, mask, image_before_resize, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    shift_val = (100 / 512) * image.shape[0]

    image_resized = cv2.resize(image_before_resize, (image.shape[1], image.shape[0]))

    mask = mask // 255
    image_matched = image * (1 - mask) + image_resized * mask
    mask = mask * 255

    img_1  = scipy.ndimage.shift(image_matched, (-shift_val, 0, 0), order=0, mode='constant', cval=1)
    mask_1  = scipy.ndimage.shift(mask, (-shift_val, 0, 0), order=0, mode='constant', cval=255)
    img_2  = scipy.ndimage.shift(image_matched, (shift_val, 0, 0), order=0, mode='constant', cval=1)
    mask_2  = scipy.ndimage.shift(mask, (shift_val, 0, 0), order=0, mode='constant', cval=255)
    img_3  = scipy.ndimage.shift(image_matched, (0, shift_val, 0), order=0, mode='constant', cval=1)
    mask_3  = scipy.ndimage.shift(mask, (0, shift_val, 0), order=0, mode='constant', cval=255)
    img_4  = scipy.ndimage.shift(image_matched, (0, -shift_val, 0), order=0, mode='constant', cval=1)
    mask_4  = scipy.ndimage.shift(mask, (0, -shift_val, 0), order=0, mode='constant', cval=255)
    image_cat = np.dstack((mask, image_matched, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4))

    mask_patch = torch.from_numpy(image_cat).to(device).float().div(255).unsqueeze(0)
    mask_patch = mask_patch.permute(0, -1, 1, 2)
    inputs = mask_patch[:, 1:, ...]
    mask = mask_patch[:, 0:1, ...]
    out = model(inputs, mask)
    out = out.mul(255).byte().data.cpu().numpy()
    out = np.transpose(out, [0, 2, 3, 1])[0]

    return out

def pad_image(image):
    x = ((image.shape[0] // 256) + (1 if image.shape[0] % 256 != 0 else 0)) * 256
    y = ((image.shape[1] // 256) + (1 if image.shape[1] % 256 != 0 else 0)) * 256
    padded = np.zeros((x, y, image.shape[2]), dtype='uint8')
    padded[:image.shape[0], :image.shape[1], ...] = image
    return padded

def main(argv):
    images_folder = '/home/abe/Workspace/dataset/arcadium2/images/'
    masks_folder = '/home/abe/Workspace/dataset/arcadium2/masks/'
    output_folder = '/home/abe/Workspace/dataset/arcadium2/inpainted_highres/'
    weights_folder = '/home/abe/Workspace/high-res-image-inpainting/weights/'
    max_size = 2048

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        opts, args = getopt.getopt(argv,"hi:m:o:w:s:",["images_folder=","masks_folder=","output_folder", "weights_folder=", "max_size="])
    except getopt.GetoptError:
        print('inpainting.py -i <images_folder> -m <masks_folder> -o <output_folder> -w <weights_folder> -s <max_size>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('inpainting.py -i <images_folder> -m <masks_folder> -o <output_folder> -w <weights_folder> -s <max_size>')
            sys.exit()
        elif opt in ("-i", "--images_folder"):
            images_folder = arg
        elif opt in ("-m", "--masks_folder"):
            masks_folder = arg
        elif opt in ("-o", "--output_folder"):
            output_folder = arg
        elif opt in ("-w", "--weights_folder"):
            weights_folder = arg
        elif opt in ("-s", "--max_size"):
            max_size = int(arg)

    for image_name in os.listdir(os.path.join(images_folder)):
        # only process valid images
        if (not image_name.lower().endswith('.jpg')) and (not image_name.lower().endswith('.png')):
            print("Not processing: ", image_name)
            continue     

        print("Processing: ", image_name)

        image_path = os.path.join(images_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        # find masks without extension
        for mask_name in os.listdir(os.path.join(masks_folder)):
            if (os.path.splitext(image_name)[0] == os.path.splitext(mask_name)[0]):
                mask_path = os.path.join(masks_folder, mask_name)
        
        # read image and mask
        img = io.imread(image_path)
        mask = io.imread(mask_path)   

        org_shape = copy.deepcopy(img.shape)  

        # downscale image since gpu memory is limited
        if (img.shape[0] > max_size or img.shape[1] > max_size):
            img = cv2.resize(img, (max_size, max_size)) 
            mask = cv2.resize(mask, (max_size, max_size))         
        
        if len(mask.shape) != 3:
            mask = mask[..., np.newaxis]

        assert img.shape[:2] == mask.shape[:2]

        mask = mask[..., :1]

        image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        shape = image.shape

        image = pad_image(image)
        mask = pad_image(mask)

        DFNet_model = DFNet().to(device)
        DFNet_model.load_state_dict(torch.load(os.path.join(weights_folder, 'model_places2.pth'), map_location=device))
        DFNet_model.eval()
        DFNET_output = preprocess_image_dfnet(image, mask, DFNet_model)

        Refinement_model = RefinementNet().to(device)
        Refinement_model.load_state_dict(torch.load(os.path.join(weights_folder, 'refinement.pth'), map_location=device)['state_dict'])
        Refinement_model.eval()

        out = preprocess_image(image, mask, DFNET_output, Refinement_model)

        out = out[:shape[0], :shape[1], ...][..., :3]
        plt.imsave(output_path, out)  

        # resizing image to original size
        output_image = cv2.imread(output_path)
        output_image = cv2.resize(output_image, (org_shape[1], org_shape[0]))
        cv2.imwrite(output_path, output_image)          

        # copying metadata
        source = Image.open(image_path)
        exif = source.getexif()
        image_new = Image.open(output_path)
        image_new.save(output_path, exif=exif)

if __name__ == "__main__":
   main(sys.argv[1:])
   

            


