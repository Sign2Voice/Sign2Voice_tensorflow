# sign2voice - tensorflow 🗣️

sign2voice is a is a team effort of Vanessa Tuerker, Bejamin Gramms, Felix Satttler, Oliver Puerner and Maximilian Scheel aiming to improve hte inclusion of people relying on sign language by providing a tool which translates **video sign langue input into audio**. 

Furhter info can be found in _sign2voice.pdf_. 

A **demo video of the streamlit app** with the tensorflow model _sign2voice_tensorflow_demo.mov_ is also attached.  

Find an outline of the model architecture below

1️⃣ SLR (sign language recognition) - recognizing glosses in video (frames)

2️⃣ G2T (gloss to text) - transforming glosses into actual sentences

3️⃣ TTS (text to speech) - creating audio output

The full pipeline is put together in a **streamlit app** which allows to take a live video which is subsequently translated into a gloss sequence, transformed into sentence(s) and finally rad out loud.

## SLR (sign language recognition) with tensorflow real time object detection

Naviage to the folder **slr_tf_rtod** and follow the following steps:

- set up & activate virtual environment **python3.10 -m venv .venv**
- install requirements  **pip install -r slr_tf_rtod_requirements.txt**
- run code/ follow instrucitons in jupyter notebooks
  - **slr_tf_rtod_create_training_data.ipynb** & 
  - **slr_tf_rtod_train_tf_model.ipynb** 

_Sample .ckpt files of the model trained with sample glosses from "montag", "auch", "mehr", "wolke", "als", "sonne", "ueberwiegend", "regen", "gewitter" from PHOENIX 2014t weather dataset to be found in _models/phoenix_new__.

**CREDITS** - the repo is largely based on Nicholas Renotte"s _Real Time Sign Language Detection with Tensorflow Object Detection and Python | Deep Learning SSD_ 

Youtube tutorial: https://www.youtube.com/watch?v=pDXdlXlaCco&ab_channel=NicholasRenotte

Github Repo: https://github.com/nicknochnack/RealTimeObjectDetection

## Gloss2Text2Speech

Prep G2T model by **adding .bin** (to be requested by the authors) in _Gloss2Text2Speech/pretrained_.

_For details how the repo works or run this model on its own check out the respective README.md file._

## Text2Speech

Prep TTS model by **create .env file i main directory** follwing this structure: 

    AZUREENDPOINT=
    APIKEY=
    AZUREDEPLOYMENT=
    APIVERSION=

_Note that the credentials the team used can unforutnately not be shared_.

_For details how the repo works or how to run this model on its own check out the respective README.md file._


# RUN the streamlit app

- create & activate virtual environment **python3.9 -m venv .venv** 
- update environment **pip install -r streamlit_requirements.txt**
- **run commands in streamlit_setup.ipynb**
- run streamlit with command **streamlit run st_to_txt/streamlit_app.py**

# FUTURE IMPROVMENTS

 - [ ] **streamlit cloud - build ready to use web/ mobile app**
 - [ ] real time object detection - switch real time object detection to pytorch as tensorflow object detection is deprecated
- [ ] vocabulary - train comprehensive model to imrpvoe generalzability & accuracy
 - [ ] TTS - evaluate free alternatives to openai

# NEXT STEPS

- [ ] update g2t & ttts readme.md Vanessa & Olli (requested in SLACK)git 