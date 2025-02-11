import sys
import streamlit as st
import os
import time
import cv2
import concurrent.futures
import tensorflow as tf
import numpy as np 
from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder 
from tensorflow.python.checkpoint.checkpoint_management import checkpoint_exists

# 🛠 Add root directories to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath("Gloss2Text2Speech"))

# ✅ Import custom functions
from gloss_to_text import gloss_to_text
from text_to_speech import text_to_speech

# 📌 Define paths
PHOENIX_MODEL_PATH = "CorrNet/pretrained_model/dev_18.90_PHOENIX14.pt"
ADAPTER_MODEL_PATH = "Gloss2Text2Speech/pretrained/adapter_model.bin"
WORKSPACE_PATH = 'slr_tf_rtod/Tensorflow/workspace'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

# Load the TensorFlow model
def load_model():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    return detection_model, detect_fn

detection_model, detect_fn = load_model()
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# 🎬 **Streamlit App Title**
st.title("Sign2Voice 🗣️ ")

# Stream live video input
st.title("🎥 Start a livestream here")

# Status variable for the stream
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# Status variable for frame count
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0  # Initialize frame_count

# Status variable for the timer
if "start_time" not in st.session_state:
    st.session_state.start_time = None  # Start time for the timer

# Status variable for the last elapsed time
if "last_elapsed_time" not in st.session_state:
    st.session_state.last_elapsed_time = None  # Last elapsed time

# Variable to store detected objects log
if "detected_objects_log" not in st.session_state:
    st.session_state.detected_objects_log = []

# Start/stop stream button
if st.button("Start Stream"):
    st.session_state.streaming = True
    st.session_state.start_time = time.time()  # Set start time
    st.session_state.last_elapsed_time = None  # Reset last elapsed time
    st.session_state.detected_objects_log = []  # Clear previous log

if st.button("Stop Stream"):
    st.session_state.streaming = False
    if st.session_state.start_time:
        # Save the elapsed time when stopping
        st.session_state.last_elapsed_time = time.time() - st.session_state.start_time
    st.session_state.start_time = None  # Reset start time

# Video capture object
cap = cv2.VideoCapture(0)  # 0 for the webcam

# Check if the video has been opened
if not cap.isOpened():
    st.error("Error accessing the webcam.")

# Create a timer placeholder that updates
timer_placeholder = st.empty()

# Define a function to log detected objects with a score threshold of 65%
def log_detected_objects(detections, category_index):
    log = []
    for i in range(detections['num_detections']):
        if detections['detection_scores'][i] >= 0.65:  # Adjusted the score threshold to 65%
            class_id = detections['detection_classes'][i]
            if class_id in category_index:
                class_name = category_index[class_id]['name']
                score = detections['detection_scores'][i]
                log.append((class_name, score))
    return log

# Function to remove consecutive duplicate glosses
def remove_consecutive_duplicates(glosses):
    unique_glosses = []
    prev_gloss = None
    for gloss in glosses:
        if gloss != prev_gloss:
            unique_glosses.append(gloss)
            prev_gloss = gloss
    return unique_glosses

# Display video if streaming is active
if st.session_state.streaming:
    stframe = st.empty()  # Placeholder for the video

    last_log_time = time.time()

    while st.session_state.streaming:
        elapsed_time = time.time() - st.session_state.start_time  # Time in seconds
        timer_placeholder.write(f"Stream running for: {elapsed_time:.2f} seconds")
        ret, frame = cap.read()
        if not ret:
            st.error("Error retrieving the video stream.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, caption="Live Video", use_container_width=True)

        # Real-time Object Detection
        input_tensor = tf.convert_to_tensor(np.expand_dims(frame_rgb, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = frame_rgb.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + 1,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.5,
            agnostic_mode=False
        )

        stframe.image(image_np_with_detections, caption="Detected Objects", use_container_width=True)

        # Log detected objects with a score of over 65%, but only once every second
        current_time = time.time()
        if current_time - last_log_time >= 1:
            logged_objects = log_detected_objects(detections, category_index)
            st.session_state.detected_objects_log.extend(logged_objects)
            last_log_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Display the detected glosses and other information after the video stops
if not st.session_state.streaming and st.session_state.detected_objects_log:
    if st.session_state.last_elapsed_time is not None:
        timer_placeholder.write(f"The stream last ran for: {st.session_state.last_elapsed_time:.2f} seconds")
    else:
        timer_placeholder.write("The stream is stopped. No time recorded.")

    st.write("🔄 Processing video")

    # Remove consecutive duplicate glosses
    unique_glosses_log = remove_consecutive_duplicates([obj[0] for obj in st.session_state.detected_objects_log])

    # Display detected glosses in a text box
    glosses_text = "\n".join(unique_glosses_log)
    st.text_area("Detected Glosses:", glosses_text, height=200)

    # Translate detected glosses into text-to-speech
    if unique_glosses_log:
        st.write("🔄 Processing glosses to text")
        generated_sentence = gloss_to_text(" ".join(unique_glosses_log))
        st.subheader("📜 Generated Sentence")
        st.write("📢", generated_sentence)
        st.write("🔄 Converting text to speech")
        text_to_speech(generated_sentence)
        st.success("✅ Text-to-Speech completed!")
