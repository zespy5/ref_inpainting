from PIL import Image
import numpy as np
import cv2
from datasets.data_utils import * 
from rembg import remove


def make_mask_images(road_mask, other_mask, num):
    
    h,w = (np.size(road_mask,0), np.size(road_mask,1))
    
    ys,xs = np.where(road_mask==255)
    positions = np.random.randint(len(xs), size=num)
    masks = []
    mask_imgs = []
    for p in positions:
        x = xs[p]
        y = ys[p]

        mask = np.zeros((h, w))
        size = 100
        left_x = max(0, x-size)
        right_x = min(x+size, w)
        up_y = max(0, y-size)
        down_y = min(y+size, h)

        mask[up_y:down_y, left_x:right_x]=255
        mask -= other_mask
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        mask_img = Image.fromarray(mask)
        masks.append(mask)
        mask_imgs.append(mask_img)
    return masks, mask_imgs

def get_seg(results, labels):
    road_mask = None
    exist_labels = {}
    for res in results:
        if res['label']=='road':
            road_mask = res['mask']
        else:
            exist_labels[res['label']] = res['mask']
    
    if road_mask==None:
        raise('there is no the road in this image')
    
    np_road_mask = np.array(road_mask)
    h,w = np.shape(np_road_mask)
    other_mask = np.zeros((h, w))
    for la in labels:
        try:
            other_mask+=exist_labels[la]
        except  KeyError:
            continue          
    
    return np_road_mask, other_mask


def make_mask_images_for_object(road_mask, object_mask, num):
    
    object_mask = (object_mask*255).astype(np.uint8)
    oh, ow = np.shape(object_mask)
    h_size = oh//2
    w_size = ow//2
    
    h,w = (np.size(road_mask,0), np.size(road_mask,1))
    
    ys,xs = np.where(road_mask==255)
    
    masks = []
    mask_imgs = []
    while len(masks)<num:
        p = np.random.randint(len(xs))
        x = xs[p]
        y = ys[p]

        if x< w_size:
            x = w_size
        if x > w-w_size:
            x = w-w_size
        if y< h_size:
            y = h_size
        if y > h-h_size:
            y = h-h_size
        mask = np.zeros((h, w))


        mask[y-h_size:y+h_size, x-w_size:x+w_size]=object_mask

        mask_img = Image.fromarray(mask.astype(np.uint8))
        masks.append(mask)
        mask_imgs.append(mask_img)
    return masks, mask_imgs


def make_mask_from_point(background, object_mask, scale, point):
    H,W = background.shape[0], background.shape[1]
    scaled_H = int(H*scale[0])
    scaled_W = int(W*scale[1])
    point_H = int(H*point[0])
    point_W = int(W*point[1])
    
    
    y1, y2, x1, x2 = get_bbox_from_mask(object_mask)
    object_mask = object_mask[y1:y2, x1:x2]
    
    object_mask = np.pad(object_mask, ((5,5),(5,5)), 'constant', constant_values=0)
    object_mask = pad_to_square2d(object_mask, pad_value=0)
    object_mask = (object_mask*255).astype(np.uint8)
    
    
    h_size = scaled_H//2
    w_size = scaled_W//2
    object_mask = cv2.resize(object_mask, (2*h_size, 2*w_size))
    

    if point_W< w_size:
        point_W = w_size
    if point_W > W-w_size:
        point_W = W-w_size
    if point_H< h_size:
        point_H = h_size
    if point_H > H-h_size:
        point_H = H-h_size
    mask = np.zeros((H, W))


    mask[point_H-h_size:point_H+h_size, point_W-w_size:point_W+w_size]=object_mask

    mask_img = Image.fromarray(mask.astype(np.uint8))

    return mask, mask_img

def generate_object_mask(input_image_path):

    input_image = Image.open(input_image_path)

    output_image = remove(input_image, only_mask=True)

    mask_array = np.array(output_image)
    mask_array[mask_array > 125] = 255  
    mask_array[mask_array <= 125] = 0
    
    mask_image = Image.fromarray(mask_array).convert("L")
    
    return mask_array, mask_image

def generate_object_img_mask(input_image):

    output_image = remove(input_image)
    
    np_out = np.array(output_image)
    img = np_out[:,:,:-1]
    mask_array = np_out[:,:,-1]
    mask_array[mask_array > 125] = 255  
    mask_array[mask_array <= 125] = 0
    
    
    return img, mask_array


