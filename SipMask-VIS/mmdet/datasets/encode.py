import os
import glob
from PIL import Image
from numpy import asarray
import numpy as np
from pycocotools import mask
import cv2
from bbox import BBox2D
import uuid

import pdb

# ann_id_file = "anns.txt" #output file that contains generated annotation IDs

def convert_binary(masks_path: str, mask_name: str):
    image = Image.open(os.path.join(masks_path, mask_name))
    data = asarray(image)
    uniqueValues = np.unique(data)

    binary_masks = []
    if len(uniqueValues) == 1:
        new = {}
        new['b_mask'] = mask.encode(np.asfortranarray(data)) # Convert to RLE to save space here 
        new['inst_id'] = None
        new['is_crowd'] = False
        new['bbox'] = None
        new['area'] = None
        new['category_id'] = None
        new['id'] = None
        new['size'] = image.size
        binary_masks.append(new) 

        return binary_masks

    for val in uniqueValues[1:]:
        new = {}
        #new['b_mask'] = mask.encode(np.asfortranarray(np.where(data != val, 0, 1)))
        temp = np.where(data != val, 0, 1).astype(np.uint8)
        new['bbox'], new['area'] = generate_bbox(temp)
        new['is_crowd'] = False
        new['b_mask'] = mask.encode(np.asfortranarray(temp))
        new['inst_id'] = val # 0 = None, 1 = Id 1, 2 = Id 2 ...
        new['category_id'] = 1 # For now it's set to always be "Person" category
        new['id'] = str(uuid.uuid4())
        new['size'] = image.size
        
        #createAnnId(new['id'])
        binary_masks.append(new)

    return binary_masks

def generate_bbox(b_mask):
    contours, hierarchy = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x, y, x+w, y+h])

    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]
    box = BBox2D((left, top, right, bottom), mode=1)
    
    return [left, top, right, bottom], box.height*box.width

# def get_ann_ids(path: str, vid: str):
#     instance_ids = []
#     ann_ids = {}
#     all_binary_masks = {}
#     '''
#     if "seg_out" in os.listdir(path):
#         masks_path = os.path.join(path, "seg_out")
#     elif "seg" in os.listdir(path):
#         masks_path = os.path.join(path, "seg")
#     '''
#     masks_path = os.path.join(path, "Annotations", vid)
#     for num_frame, frame in enumerate(sorted(os.listdir(masks_path))):
#         if frame.endswith((".png")): binary_masks = convert_binary(masks_path, frame)
#         else: continue
#         id_root = vid + '/' + frame
#         ann_ids[id_root] = []
        
#         for mask in binary_masks:
#             if not mask['inst_id']:
#                 continue
#             elif mask['inst_id'] not in instance_ids:
#                 instance_ids.append(mask['inst_id'])
#                 all_binary_masks[mask['inst_id']] = []
#             all_binary_masks[mask['inst_id']].append({'mask_data':{'mask': mask['b_mask'], 'bbox': mask['bbox'], \
#                                                                    'area': mask['area'], 'cat_id': mask['category_id'], \
#                                                                   'is_crowd': mask['is_crowd'], 'id': mask['id']}, \
#                                                       'num_frame': num_frame})
#             ann_ids[id_root].append(mask['id'])
        
#     # Generate unique annotation Ids
#     #generateAnnIds(ann_id_file, ann_ids)
    
#     return all_binary_masks, instance_ids

def get_anns(path, vid):
    instance_ids = []
#     ann_ids = {}
#     all_binary_masks = {}
    ann_info = dict()
    '''
    if "seg_out" in os.listdir(path):
        masks_path = os.path.join(path, "seg_out")
    elif "seg" in os.listdir(path):
        masks_path = os.path.join(path, "seg")
    '''
    masks_path = os.path.join(path, "Annotations", vid)
    frames_list = sorted(glob.glob(os.path.join(masks_path, "*.png")))
    total_frames = len(frames_list)
    for num_frame, frame in enumerate(frames_list):
        binary_masks = convert_binary(masks_path, frame)
#         if frame.endswith((".png")): binary_masks = convert_binary(masks_path, frame)
#         else: continue
#         id_root = vid + '/' + frame
#         ann_ids[id_root] = []
        
        for mask in binary_masks:
            if not mask['inst_id']:
                continue
            elif mask['inst_id'] not in instance_ids:
                ann_info[mask['inst_id']] = [None] * total_frames
                instance_ids.append(mask['inst_id'])
#                 all_binary_masks[mask['inst_id']] = []
            ann_info[mask['inst_id']][num_frame] =  {'mask': mask['b_mask'], 'bbox': mask['bbox'], \
                                                     'area': mask['area'], 'cat_id': mask['category_id'], \
                                                     'is_crowd': mask['is_crowd'], 'id': mask['id'], \
                                                     'size': mask['size']}
#             all_binary_masks[mask['inst_id']].append({'mask_data': {'mask': mask['b_mask'], 'bbox': mask['bbox'], \
#                                                                    'area': mask['area'], 'cat_id': mask['category_id'], \
#                                                                   'is_crowd': mask['is_crowd'], 'id': mask['id']}, \
#                                                       'num_frame': num_frame})
#             ann_ids[id_root].append(mask['id'])    
#     all_binary_masks, instance_ids = get_ann_ids(path, vid)
#     ann_info = {instance_id:[] for instance_id in instance_ids}
    '''
    if "seg_out" in os.listdir(path):
        masks_path = os.path.join(path, "seg_out")
    elif "seg" in os.listdir(path):
        masks_path = os.path.join(path, "seg")
    '''
#     masks_path = os.path.join(path, "Annotations", vid)
#     total_frames = len([file for file in frames_list if file.endswith(".png")])
    
#     for id_ in instance_ids:
#         ann_info[id_] = [None] * total_frames
#         for i in all_binary_masks[id_]:
#             frame_num = i['num_frame']
#             ann_info[id_][frame_num] = i['mask_data']
    return ann_info

def generateAnnIds(file, ann_ids):
    f = open(file, "a+")
    for key, value in ann_ids.items():
        for mask_id in value:
            f.write(key + ': ' + mask_id + "\n")
    f.close()

def check_mask(name: str):
    image = Image.open(name)
    data = asarray(image)
    print(data.shape)
    uniqueValues = np.unique(data)
    #print(uniqueValues)
    return uniqueValues


#masks = sorted([im for im in os.listdir(MASKS_PATH)])
#imgs = sorted([im for im in os.listdir(IMGS_PATH)])

'''
    for image in os.listdir(masks_path):
        binary_masks = convert_binary(masks_path, image)
        
        for mask in binary_masks:
            if mask[inst_id] not in binary_masks.keys:
                binary_masks[mask[inst_id]] = [mask[b_mask]]
            else 
                binary_masks[mask[inst_id]].append(mask[b_mask])
'''


'''
image = Image.open(os.path.join(MASKS_PATH, '_s258.png'))
data = asarray(image)
uniqueValues = np.unique(data)

print(uniqueValues)
'''
'''
listing = []
for mask in masks:
    image = Image.open(os.path.join(MASKS_PATH, mask))
    data = asarray(image)
    uniqueValues = np.unique(data)
    listing.append(uniqueValues)
    #print(uniqueValues)
    #if np.array_equal(uniqueValues, np.array([0,2])):
    #    listing.append(mask)
    #    print(uniqueValues)

#print(listing)
'''

'''
#print(os.path.exists('/n/pfister_lab2/Lab/vcg_natural/YouTop-VIS/comedy/qVMW_1aZXRk/seg_out/o.vsseg_export_s316.png'))
mapping = {}
for im, mask in zip(imgs, masks):
    mapping[im] = mask

#print(mapping('image_14851.png'))
print(masks[0], imgs[0])
print(len(masks), len(imgs))
'''