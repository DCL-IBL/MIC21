import json
import os
import cv2
import argparse

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Convert image directory to Yolact dataset')
    parser.add_argument('--idir',
                        default='.', type=str,
                        help='Directory where are stored images.')
    parser.add_argument('--outfile',
                        default='info.json', type=str,
                        help='Output JSON description.')
    global args
    args = parser.parse_args(argv)

if __name__ == '__main__':
    parse_args()
    onlyfiles = [f for f in os.listdir(args.idir) if os.path.isfile(os.path.join(args.idir, f))]
    dataset = {}
    dataset['info'] = {'year': 2021,'version': "11",'description': args.idir,'contributor': 'jordan'}
    dataset['annotations'] = []
    dataset['images'] = []
    for (k,f) in enumerate(onlyfiles):
        filename, file_extension = os.path.splitext(f)
        if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.png':
            print(f)
            img = cv2.imread(os.path.join(args.idir, f))
            if img is not None:
                if len(img.shape) < 3:
                    continue
                height, width, channels = img.shape
                if channels != 3:
                    continue
                dataset['images'].append({'id': k,'width': width,'height': height,'file_name': f})
    with open(args.outfile, 'w') as outfile:
        json.dump(dataset, outfile)
        

    
