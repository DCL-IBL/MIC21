from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer

import numpy as np
import os
import json
import cv2
from pycocotools import mask

def polyFromMask(masked_arr):
    contours,_ = cv2.findContours(masked_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    return segmentation

def prepare_detectron2_predictor(th):
    cfg = get_cfg()
    cfg.merge_from_file('/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = th
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = th
    return DefaultPredictor(cfg)

def prepare_mic21_predictor(th,model_name):
    gt_c = json.load(open('/host/mic21-framework/server/'+model_name+'_gt.json','r'))
    cfg = get_cfg()
    cfg.merge_from_file('/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml')
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = th
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = th
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(gt_c['categories'])
    cfg.MODEL.DEVICE='cpu'
    pred = DefaultPredictor(cfg)
    checkpointer = DetectionCheckpointer(pred.model)
    try:
        checkpointer.load('/host/mic21-framework/work/output/'+model_name+'.pth')
        return pred
    except:
        return None

def prediction_with_detectron2(idir,pred,fname):
        out_json = {'images':[],'annotations':[],'categories':[]}
        cats = MetadataCatalog.get('coco_2017_train').thing_classes
        for (k,ct) in enumerate(cats):
            out_json['categories'].append({'id':k+1,'name':ct})

        #for im in gt['images']:
            #print(im['file_name'])
            #gt_imgs[os.path.basename(im['path'])] = im['id']
        #    gt_imgs[im['file_name']] = im['id']
            
        aind = 0
        img_id = 0
        for (k,f) in enumerate(os.listdir(idir)):
            fn, fext = os.path.splitext(f)
            if fext != '.jpg' and fext != '.jpeg' and fext != '.png':
                continue
            full_name = os.path.join(idir,f)
            try:
                img = read_image(full_name)
            except:
                continue
            if len(img.shape) < 3:
                continue
            if img.shape[2] != 3:
                continue
            print(f)
            prediction = pred(img)
            h,w = prediction['instances'].image_size
            out_json['images'].append({'file_name':f,'height':h,'width':w,'id':img_id})
            for (ci,c) in enumerate(prediction['instances'].pred_classes):
                msk = mask.encode(np.asfortranarray(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)))
                msk['counts'] = msk['counts'].decode('ascii')
                bboxes = prediction['instances'].pred_boxes.tensor.cpu().numpy().tolist()
                bb = bboxes[ci]
                bb[2] = bb[2] - bb[0]
                bb[3] = bb[3] - bb[1]
                out_json['annotations'].append({'id':aind,
                                                'category_id': c.cpu().item()+1, #gt_cats[cats[c.cpu().item()]],
                                                #'category_id':c.cpu().item(),
                                                'image_id':img_id,                                               #'segmentation':polyFromMask(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)),
                                                'segmentation':msk,
                                                'bbox':bb,
                                                'score':prediction['instances'].scores[ci].cpu().item(),
                                                'iscrowd':1,                                                'area':np.sum(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)).astype(np.float)})
                aind = aind + 1
            img_id = img_id + 1

        fout = open(fname,'w')
        fout.write(json.dumps(out_json))
        fout.close()

def prediction_with_mic21(folder_name,pred,fname):
    idir = '/host/mic21-framework/server/uploads/'+folder_name+'/'
    gt_cats = dict()
    
    out_json = {'images':[],'annotations':[],'categories':[]}
    gt_c = json.load(open('/host/mic21-framework/server/'+folder_name+'_gt.json','r'))
    for (k,c) in enumerate(gt_c['categories']):
        out_json['categories'].append({'id':k,'name':c['name']})
            
    aind = 0
    img_id = 0
    for (k,f) in enumerate(os.listdir(idir)):
        fn, fext = os.path.splitext(f)
        if fext != '.jpg' and fext != '.jpeg' and fext != '.png':
            continue
        print(f)
        full_name = os.path.join(idir,f)
        try:
            img = read_image(full_name)
        except:
            continue
        if len(img.shape) < 3:
            continue
        if img.shape[2] != 3:
            continue
        prediction = pred(img)
        h,w = prediction['instances'].image_size
        out_json['images'].append({'file_name':f,'height':h,'width':w,'id':img_id})
        for (ci,c) in enumerate(prediction['instances'].pred_classes):
            msk = mask.encode(np.asfortranarray(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)))
            msk['counts'] = msk['counts'].decode('ascii')
            bboxes = prediction['instances'].pred_boxes.tensor.cpu().numpy().tolist()
            bb = bboxes[ci]
            bb[2] = bb[2] - bb[0]
            bb[3] = bb[3] - bb[1]
            out_json['annotations'].append({'id':aind,
                                            #'category_id':gt_cats[cats[c.cpu().item()]],
                                            'category_id':c.cpu().item(),
                                            'image_id':img_id,                                               #'segmentation':polyFromMask(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)),
                                            'segmentation':msk,
                                            'bbox':bb,
                                            'score':prediction['instances'].scores[ci].cpu().item(),
                                            'iscrowd':1,                                  'area':np.sum(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)).astype(np.float)})
            aind = aind + 1
        img_id = img_id + 1

    fout = open(fname,'w')
    fout.write(json.dumps(out_json))
    fout.close()

def prediction_with_detectron2_single(full_name,pred,fname):
    out_json = {'images':[],'annotations':[],'categories':[]}
    cats = MetadataCatalog.get('coco_2017_train').thing_classes
    for (k,ct) in enumerate(cats):
        out_json['categories'].append({'id':k+1,'name':ct})
            
    aind = 0
    img_id = 0
    img = read_image(full_name)
    print(full_name)
    prediction = pred(img)
    h,w = prediction['instances'].image_size
    out_json['images'].append({'file_name':os.path.basename(full_name),'height':h,'width':w,'id':img_id})
    for (ci,c) in enumerate(prediction['instances'].pred_classes):
        msk = mask.encode(np.asfortranarray(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)))
        msk['counts'] = msk['counts'].decode('ascii')
        bboxes = prediction['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        bb = bboxes[ci]
        bb[2] = bb[2] - bb[0]
        bb[3] = bb[3] - bb[1]
        out_json['annotations'].append({'id':aind,
                                        'category_id': c.cpu().item()+1, #gt_cats[cats[c.cpu().item()]],
                                        'image_id':img_id,                                               
                                        'segmentation':msk,
                                        'bbox':bb,
                                        'score':prediction['instances'].scores[ci].cpu().item(),
                                        'iscrowd':1,                                                'area':np.sum(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)).astype(np.float)})
        aind = aind + 1
                
    fout = open(fname,'w')
    fout.write(json.dumps(out_json))
    fout.close()

def prediction_with_mic21_single(full_name,folder_name,pred,fname):
    gt_cats = dict()
    
    out_json = {'images':[],'annotations':[],'categories':[]}
    gt_c = json.load(open('/host/mic21-framework/server/'+folder_name+'_gt.json','r'))
    for (k,c) in enumerate(gt_c['categories']):
        out_json['categories'].append({'id':k,'name':c['name']})
            
    aind = 0
    img_id = 0
    img = read_image(full_name)
    prediction = pred(img)
    h,w = prediction['instances'].image_size
    out_json['images'].append({'file_name':os.path.basename(full_name),'height':h,'width':w,'id':img_id})
    for (ci,c) in enumerate(prediction['instances'].pred_classes):
        msk = mask.encode(np.asfortranarray(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)))
        msk['counts'] = msk['counts'].decode('ascii')
        bboxes = prediction['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        bb = bboxes[ci]
        bb[2] = bb[2] - bb[0]
        bb[3] = bb[3] - bb[1]
        out_json['annotations'].append({'id':aind,
                                        'category_id':c.cpu().item(),
                                        'image_id':img_id,                                              
                                        'segmentation':msk,
                                        'bbox':bb,
                                        'score':prediction['instances'].scores[ci].cpu().item(),
                                        'iscrowd':1,                                  'area':np.sum(prediction['instances'].pred_masks[ci,:,:].cpu().numpy().astype(np.uint8)).astype(np.float)})
        aind = aind + 1
        
    fout = open(fname,'w')
    fout.write(json.dumps(out_json))
    fout.close()