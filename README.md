# movements-and-facial-detection

This is a project developed for academic purposes.  
The project performs facial detection and emotional expression analysis, activity detection and categorization, and generates an automatic summary in .txt format.  
</br>
The emotions addressed are:

- Happy
- Sad
- Neutral
- Fear
- Angry
- Surprise

The activities considered for categorization are:

- Head
  - Neutral
  - Right
  - Left
  - Up
  - Down
  - Tilted
- Hands (both)
  - Neutral
  - Raised
  - Moving
- Arms (both)
  - Neutral
  - Up
  - Extended

The project was developed in Python using the following libraries:

```
opencv-python
mediapipe
tqdm
deepface
tf_keras
```

## Features

1. Facial recognition: identification and marking of faces present in the video.
2. Analysis of emotional expressions of the identified faces.
3. Detection and categorization of activities performed in the video.
4. Automatic summary generation of the main activities and emotions detected in the video.

## Notes

- All input and output files for the project, including the video for analysis, generated .mp4, and .txt files, are located in `app\src\resources`.

## How to configure and run the project

```bash
$ python --version
Python 3.12.4
```

### Step-by-step (Windows)

The entire project was developed using Python's virtual environment. Follow the steps below to execute it on your machine:

1. Download and install **Python 3.12.4** from [python.org/downloads/python-3124](https://www.python.org/downloads/release/python-3124/).
2. Clone this project into a folder on your Windows system ([docs.github.com/cloning-a-repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository#cloning-a-repository)).
3. Navigate to the project's root directory and enter the following commands to create, configure, and activate the Python virtual environment:
    - `$ cd app`
    - `$ python -m venv .venv`
    - `$ source .venv/Scripts/activate` for Bash or `.venv\Scripts\activate.bat` for CMD
    - `$ pip install -r requirements.txt`

Once the configurations are complete, the project will be ready to run on your machine.  
While still in the `app` folder, to execute the project, simply enter the command:

```bash
$ python src/main.py
...
```

Monitor the execution progress through the console and check the report generated in `app\src\resources`.
