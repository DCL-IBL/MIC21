import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
import json
import fiftyone as fo
import fo_utils
import det2_utils

UPLOAD_PATH = '/host/app1/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__,static_folder='')

# Serve React App
#@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    return send_from_directory(app.static_folder, path)
#    if path != "" and os.path.exists(app.static_folder + '/' + path):
#        return send_from_directory(app.static_folder, path)
#    else:
#        return send_from_directory(app.static_folder, 'index.html')

api = Api(app)
app.config['UPLOAD_PATH'] = UPLOAD_PATH

#class Annotate(Resource):
    
#class AnnotateYolact(Annotate):
    
#class AnnotateDetectron2(Annotate):

#class TrainMIC21_Model(Resource):

#class AnnotateMIC21(Annotate):

#class HelloWorld(Resource):
#    def get(self):
#        return {'hello': 'world'}
    
#api.add_resource(HelloWorld, '/')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        categ = request.form['categ_name']
        folder_path = '/uploads/'+categ+'/'
        full_path = '/host/app1/server'+folder_path
        if not os.path.exists(full_path):
            os.mkdir(full_path)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_name = os.path.join(full_path, filename)
            print(full_name)
            file.save(full_name)
            
            dataset = fo.load_dataset(categ)
            dataset.add_sample(fo.Sample(filepath=full_name))
            out_name = '/host/app1/server/uploads/buf.json'
            
            os.system("rm /host/dir1/image_out/*.*")
            os.system("cp "+full_name+" /host/dir1/image_out/")
            os.system("python3 /host/dir1/prepare_dataset.py --idir=/host/dir1/image_out --outfile=/host/dir1/info.json")
            os.system("python3 /host/dir1/yolact/eval.py --trained_model=/host/dir1/yolact/yolact_base_54_800000.pth --score_threshold=0.9 --output_web_json --dataset=test1 --cuda=0 --web_det_path=/host/dir1/yolact/web/dets")
            os.system("mv /host/dir1/yolact/web/dets/yolact_base.json "+out_name)
            fo_utils.create_prediction(dataset,full_path,'yolact',out_name)
            
            pred = det2_utils.prepare_detectron2_predictor(0.9)
            det2_utils.prediction_with_detectron2_single(full_name,pred,out_name)
            fo_utils.create_prediction(dataset,full_path,'detectron2',out_name)
            
            pred = det2_utils.prepare_mic21_predictor(0.9,categ)
            det2_utils.prediction_with_mic21_single(full_name,categ,pred,out_name)
            fo_utils.create_prediction(dataset,full_path,'mic21',out_name)
            
    return redirect("http://dcl.bas.bg:1317")
    return '''
    <head>
    <title>Upload new File</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>
    <script src="/upload_images.js"></script>
    </head>
    <body onload="init_data()">
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <label for="top_categ">Top level category:</label>
      <select name="top_categ" id="top_categ"></select><br>
      <div name="form1" id="form1"></div><br>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    <div name="image_container" id="image_container"></div>
    </body>
    '''


@app.route('/load_dataset/', methods=['GET'])
def load_dataset():
    if request.method == 'GET':
        categ = request.args.get("categ_name")
        folder_path = '/uploads/'+categ+'/'
        full_path = '/host/app1/server'+folder_path
        try:
            dataset = fo.load_dataset(categ)
            dataset.delete()
        except:
            print("New dataset")
        dataset = fo.Dataset.from_dir(dataset_dir=full_path, dataset_type=fo.types.ImageDirectory, name=categ)
        dataset.persistent = True
        yolact_name = '/host/app1/server/uploads/'+categ+'_yolact.json'
        detectron2_name = '/host/app1/server/uploads/'+categ+'_detectron2.json'
        mic21_name = '/host/app1/server/uploads/'+categ+'_mic21.json'
        gtruth_name = '/host/app1/server/uploads/'+categ+'_gt.json'
        if os.path.exists(yolact_name):
            fo_utils.create_prediction(dataset,full_path,'yolact',yolact_name)
        if os.path.exists(detectron2_name):
            fo_utils.create_prediction(dataset,full_path,'detectron2',detectron2_name)
        if os.path.exists(mic21_name):
            fo_utils.create_prediction(dataset,full_path,'mic21',mic21_name)
        if os.path.exists(gtruth_name):
            fo_utils.create_annotation(dataset,full_path,'ground_truth',gtruth_name)
        return "<html><head></head><body></body></html>"

@app.route('/predict/', methods=['GET'])
def predict():
    links = []
    if request.method == 'GET':
        model = request.args.get("model")
        categ = request.args.get("categ_name")
        threshold = float(request.args.get("threshold"))
        folder_path = '/uploads/'+categ+'/'
        full_path = '/host/app1/server'+folder_path
        if model == 'yolact':
            os.system("rm /host/dir1/image_out/*.*")
            os.system("cp "+full_path+"*.* /host/dir1/image_out/")
            os.system("python3 /host/dir1/prepare_dataset.py --idir=/host/dir1/image_out --outfile=/host/dir1/info.json")
            os.system("python3 /host/dir1/yolact/eval.py --trained_model=/host/dir1/yolact/yolact_base_54_800000.pth --score_threshold="+str(threshold)+" --output_web_json --dataset=test1 --cuda=0 --web_det_path=/host/dir1/yolact/web/dets")
            os.system("mv /host/dir1/yolact/web/dets/yolact_base.json /host/app1/server/uploads/"+categ+"_yolact.json")
        if model == 'detectron2':
            pred = det2_utils.prepare_detectron2_predictor(threshold)
            det2_utils.prediction_with_detectron2("/host/app1/server/uploads/"+categ,
                                                  pred,
                                                  "/host/app1/server/uploads/"+categ+"_detectron2.json")
        if model == 'mic21':
            pred = det2_utils.prepare_mic21_predictor(threshold,categ)
            print(pred)
            det2_utils.prediction_with_mic21(categ,pred,"/host/app1/server/uploads/"+categ+"_mic21.json")
        return "<html><head></head><body></body></html>"
            

@app.route('/show/', methods=['GET'])
def show():
    links = []
    if request.method == 'GET':
        categ = request.args.get("categ_name")
        folder_path = '/uploads/'+categ+'/'
        full_path = '/host/app1/server'+folder_path
        return "<html><head></head><body></body></html>"
        
@app.route('/get_image_links/', methods=['GET'])
def get_image_links():
    links = []
    if request.method == 'GET':
        categ = request.args.get("categ_name")
        folder_path = '/uploads/'+categ+'/'
        full_path = '/host/app1/server'+folder_path
        app.config['UPLOAD_PATH'] = full_path
        if os.path.exists(full_path):
            for f in os.listdir(full_path):
                if allowed_file(f):
                    links.append(folder_path+f)
        else:
            os.mkdir(full_path)
        return json.dumps(links)

if __name__ == '__main__':
    #ses = fo.launch_app(remote=True)
    app.run(host='0.0.0.0',debug=True,use_reloader=True,threaded=True)