import fiftyone as fo
import glob
from pycocotools.coco import COCO
import os
import numpy as np
import json
import pycocotools.mask as maskUtils

def create_annotation(dataset,img_path='/host/comparison/acrobatics/data',name='ground_truth',coco_data='/host/comparison/acrobatics/labels.json'):
    dat_src = COCO(coco_data)

    samples = []
    for ind in dat_src.anns:
        ann = dat_src.anns[ind] # get annotation by index
        if ann['category_id'] not in dat_src.cats.keys():
            continue
        img_id = ann['image_id'] # get image_id for that annotation
        if img_id not in dat_src.imgs.keys():
            print(str(img_id) + ' not found')
            continue
        img = dat_src.imgs[img_id] # get image by image_id
        fname = os.path.join(img_path,img['file_name'])
        try:
            sample = dataset[fname] # math file name
        except:
            print(fname + ' not found')
            continue
        
        if name not in sample.field_names:
            sample[name] = fo.Detections()
        if sample[name] == None:
            sample[name] = fo.Detections()
            
        if 'score' not in ann.keys():
            conf = 1.0
        else:
            conf = ann['score']
        try:
            det = fo.Detection(
                label=dat_src.cats[ann['category_id']]['name'],
                bounding_box=ann['bbox'],
                mask=dat_src.annToMask(ann),
                confidence=conf
            )
        except:
            continue
        det.bounding_box[0] = det.bounding_box[0]/img['width']
        det.bounding_box[2] = det.bounding_box[2]/img['width']
        det.bounding_box[1] = det.bounding_box[1]/img['height']
        det.bounding_box[3] = det.bounding_box[3]/img['height']
        #print(det.mask.shape)
        #print(ann['bbox'])
        det.mask = det.mask[int(ann['bbox'][1]):int(ann['bbox'][1]+ann['bbox'][3]),int(ann['bbox'][0]):int(ann['bbox'][0]+ann['bbox'][2])]
        #print(det.mask.shape)
        #return
        sample[name]['detections'].append(det)
        sample.save()

def create_prediction(dataset,img_path,name,coco_pred):
    ann_pred = json.load(open(coco_pred,'r'))
    
    for ann in ann_pred['annotations']:
        img_id = ann['image_id'] # get image_id for that annotation
        img_fn = ''
        img_h = 0
        img_w = 0
        for img_rec in ann_pred['images']:
            if img_id == img_rec['id']:
                img_fn = img_rec['file_name']
                if 'height' in img_rec.keys():
                    img_h = img_rec['height']
                    img_w = img_rec['width']
                break
        if img_fn == '':
            continue
        fname = os.path.join(img_path,img_fn) #os.path.join(img_path,img['file_name'])
        try:
            sample = dataset[fname] # math file name
        except:
            print(fname+' not found')
            continue
        
        if name not in sample.field_names:
            sample[name] = fo.Detections()
        if sample[name] == None:
            sample[name] = fo.Detections()
            
        if 'score' not in ann.keys():
            conf = 1.0
        else:
            conf = ann['score']
        
        clabel = ''
        for ct in ann_pred['categories']:
            if ct['id'] == ann['category_id']:
                clabel = ct['name']
                break
        if clabel == '':
            print('Category not found')
            continue
        
        det = fo.Detection(
            label=clabel,
            bounding_box=ann['bbox'],
            #mask=dat_src.annToMask(ann),
            mask = maskUtils.decode(ann['segmentation']),
            confidence=conf
        )
        if img_h == 0:
            img_h = ann['segmentation']['size'][0]
            img_w = ann['segmentation']['size'][1]
        det.bounding_box[0] = det.bounding_box[0]/img_w
        det.bounding_box[2] = det.bounding_box[2]/img_w
        det.bounding_box[1] = det.bounding_box[1]/img_h
        det.bounding_box[3] = det.bounding_box[3]/img_h
        det.mask = det.mask[int(ann['bbox'][1]):int(ann['bbox'][1]+ann['bbox'][3]),int(ann['bbox'][0]):int(ann['bbox'][0]+ann['bbox'][2])]
        sample[name]['detections'].append(det)
        sample.save()