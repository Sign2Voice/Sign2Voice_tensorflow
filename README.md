# sign2voice 🗣️ - tensorflow 

sign2voice is aiming to improve the inclusion of people relying on sign language by providing a tool which translates **video sign language input into audio**. 

Further details about the project and the team can be found in _sign2voice.pdf_.

Below the **model architecture** is outlined:

1️⃣ SLR (sign language recognition) - recognizing glosses in live video input

2️⃣ G2T (gloss to text) - transforming glosses into actual text incl. grammar

3️⃣ TTS (text to speech) - transforming text into audio

The full pipeline is put together in a **streamlit web app** which allows you to take a live video which is then translated into a gloss sequence, subsequently transformed into sentence(s) and finally read out loud.

A **demo video of the MVP using tensorflow real time object detection** built in streamlit can be found here: 

[![Watch the video](https://drive.google.com/file/d/1JSBr99EUXGYUFd2FPKS_RqxbFEJuexeA/view?usp=drive_link/maxresdefault.jpg)](https://drive.google.com/file/d/1JSBr99EUXGYUFd2FPKS_RqxbFEJuexeA/view?usp=drive_link)

## SLR (sign language recognition) with tensorflow real time object detection

Navigate to the folder **slr_tf_rtod** and follow these steps:

- set up & activate virtual environment `python3.10 -m venv .venv`
- install requirements  `pip install -r slr_tf_rtod_requirements.txt`
- run code in jupyter notebooks
  - _slr_tf_rtod_create_training_data.ipynb_ & 
  - _slr_tf_rtod_train_tf_model.ipynb_

Sample .ckpt files for the tensorflow model trained with the sample glosses _"montag", "auch", "mehr", "wolke", "als", "sonne", "ueberwiegend", "regen", "gewitter"_ from _PHOENIX 2014t weather data_ can be found in _slr_tf_rtod/Tensorflow/workspace/models/phoenix_new_.

**CREDITS** - the repo is largely based on Nicholas Renotte's _Real Time Sign Language Detection with Tensorflow Object Detection and Python | Deep Learning SSD_.

Youtube tutorial: https://www.youtube.com/watch?v=pDXdlXlaCco&ab_channel=NicholasRenotte

Github Repo: https://github.com/nicknochnack/RealTimeObjectDetection

## Gloss2Text2Speech

Get the G2T model ready by adding the _adapter_model.bin_ file (to be requested with the authors) in the _Gloss2Text2Speech/pretrained_ folder. 

For details on how the model works check out the respective README.md file in _Gloss2Text2Speech_.

## Text2Speech

Get the TTS model ready by creating a _.env_ file in the repo with the following structure:

    AZUREENDPOINT=
    APIKEY=
    AZUREDEPLOYMENT=
    APIVERSION=

_Note that the credentials used by the team cannot be shared externally._ 

For details on how the model works check out the respective README.md file in _Gloss2Text2Speech_.


# RUN the streamlit app

- create & activate virtual environment `python3.9 -m venv .venv`
- update environment `pip install -r streamlit_requirements.txt`
- run commands in jupyter notebook _streamlit_setup.ipynb_
- run streamlit web app with `streamlit run st_to_txt/streamlit_app.py`

# FUTURE IMPROVEMENTS

 - [ ] streamlit cloud - build ready to use web/ mobile app
 - [ ] real time object detection - switch real time object detection to pytorch as tensorflow object detection is deprecated
- [ ] vocabulary - train comprehensive model to improve generalizability & accuracy of gloss detection
 - [ ] TTS - evaluate free alternatives to OpenAI TTS solution currently used