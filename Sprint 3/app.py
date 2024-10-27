import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
paths = {
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
}

files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], 'label_map.pbtxt')
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-9')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# Initialize the camera
cap = cv2.VideoCapture(0)
detected_sign = ""

def generate_frames():
    global detected_sign
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            image_np = np.array(frame)

            # Perform detection
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # Visualize detection
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

            # Get the most likely detection class
            if len(detections['detection_classes']) > 0:
                detected_class = detections['detection_classes'][0]
                detected_sign = category_index[detected_class + 1]['name'] if detected_class in category_index else "Unknown"

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
            frame = buffer.tobytes()

            # Yield the frame as part of the HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sign')
def get_sign():
    return jsonify({'detected_sign': detected_sign})

if __name__ == '__main__':
    app.run(debug=True)
