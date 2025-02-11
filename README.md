# sign2voice - tensorflow 🗣️

sign2voice is a is a team effort of Vanessa Tuerker, Bejamin Gramms, Felix Satttler, Oliver Puerner and Maximilian Scheel aiming to improve hte inclusion of people relying on sign language by providing a tool which translates **video sign langue input into audio**. 

Find an outline of the model architecture below

1️⃣ SLR (sign language recognition) - recognizing glosses in video (rame)

2️⃣ G2T (gloss to text) - transforming glosses into actual sentences

3️⃣ TTS (text to speech) - reading out sentences

The full pipeline is put together in a **streamlit app** which allows to take a live video which is subsequently translated into a gloss sequence which in turn is transformed into sentence(s) and finally outputted as audio. 

## SLR (sign language recognition) with tensorflow real time object detection

Naviage to the folder **slr_tf_rtod** and follow the following steps:

- set virtual environment **python3.10 -m venv .venv**
- activate virtual envionment **source .venv/bin/activate**
- install requirements  **pip install -r slr_tf_rtod_requirements.txt**
- run **slr_tf_rtod_create_training_data.ipynb** - create labeled images as training data for predefined tensorflow CNN (convoutioanl neural network) following the instructions in the script


- run **slr_tf_rtod_train_tf_model.ipynb** - train the model follwing the instructinons in the script

_Sample .ckpt files of the model trained with sample glosses from PHOENIX 2014t weather dataset to be found in _models/phoenix_new__.

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

_For details how the repo works or run this model on its own check out the respective README.md file._


# RUN the streamlit app

- create new **python3.9 -m venv .venv** 
- activate virtual envionment **source .venv/bin/activate**
- update environment **pip install -r streamlit_requirements.txt**
- **run commands in streamlit_setup.ipynb**
- run streamlit with command **streamlit run st_to_txt/streamlit_app.py**

# FUTURE IMPROVMENTS

 - [ ] **streamlit cloud - build ready to use web/ mobile app**
 - [ ] pytorch & phoenix weather data - train comprehensive model to imrpvoe generalzability & accuracy
 - [ ] TTS free alternatives

# NEXT STEPS

- [ ] finalze repo
  - [ ] **move repo to sign2voice_rtod to sign2voice github account** with name sign2voice_tensorflow & fork to private account
  - [ ] **delete maximilianbenjaminscheel organization**
  - [ ] include slide as pdf (remove neuefische logo & slides tbd)
  - [ ] use & fianlize updated requirements plus update/ remove any update code not needed anymore
  - [ ] check/ update gitignore files (exclude .bin g2t provided by author & credentials open ai)
  - [ ] update g2t & ttts readme.md Vanessa & Olli (requested in SLACK)