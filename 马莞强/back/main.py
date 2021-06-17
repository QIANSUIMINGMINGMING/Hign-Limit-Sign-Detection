import os
import sys
import base64
import shutil
import cv2
from PIL import Image
from yolo import darknet
from yolo import classification
from torch_faster import faster_rcnn
from number import number
from flask import Flask, jsonify, make_response, abort, request
os.chdir(sys.path[0])

class_names = []
with open("yolo/obj.names", "r") as f:
    for line in f:
        if len(line) > 0:
            class_names.append(line)
net = darknet.load_net(b"yolo/yolo-obj.cfg", b"yolo/yolo-obj_final.weights", 0)

app = Flask(__name__)

def wrapResponse(data):
    response = make_response(jsonify(data))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/get-image-list')
def getImageList():
    return wrapResponse(os.listdir("./images"))

@app.route('/get-image/<string:filename>')
def getImage(filename):
    if not os.path.exists("images/" + filename):
        abort("404")
    with open("images/" + filename, "rb") as f:
        return f.read()

@app.route("/upload-image/<string:model>", methods=['POST'])
def uploadImage(model):
    img = request.files.get('file')
    if os.path.exists("images/" + img.filename):
        abort("404")
    img.save("images/" + img.filename)

    bboxes = []
    cv2_image = cv2.imread("images/" + img.filename, 1)
    cropped = None
    if model == "yolov4":
        darknetImage = darknet.load_image(("images/" + img.filename).encode(), 0, 0)
        results = darknet.detect_image(net, class_names, darknetImage, thresh=.2)
        for result in results:
            bboxes.append([int(result[2][0]), int(result[2][1]),int(result[2][2]),int(result[2][3])])
    elif model == "faster rcnn":
        bboxes = faster_rcnn.detect(cv2_image)
        bboxes = [[(bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2, bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in bboxes]
    
    for bbox in bboxes:
        cropped = cv2_image[bbox[1] - bbox[3]//2:bbox[1] + bbox[3]//2, bbox[0] - bbox[2]//2:bbox[0] + bbox[2]//2]
        if classification.classify(Image.fromarray(cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB))):
            cv2.rectangle(cv2_image, (bbox[0] - bbox[2]//2-3, bbox[1] - bbox[3]//2-3), (bbox[0] + bbox[2]//2+3, bbox[1] + bbox[3]//2+3), (0, 255, 255), 2)
            height = number.give_result(cropped)
            cv2.putText(cv2_image, str(height),  (bbox[0] - bbox[2]//2, bbox[1] - bbox[3]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1, cv2.LINE_AA)
    cv2_image = cv2.resize(cv2_image, (512, 512))
    cv2.imwrite("images/" + img.filename, cv2_image)
    return wrapResponse({"filename":img.filename})

@app.route('/delete-image/<int:index>/<string:filename>')
def deleteImage(index, filename):
    if not os.path.exists("images/" + filename):
        abort("404")
    os.remove("images/" + filename)
    return wrapResponse({"index":index})

@app.route("/clear")
def clear():
    shutil.rmtree("images")
    os.mkdir("images")
    return wrapResponse({"ok":"ok"})