# Generating Talking Face Landmarks

The code for [the paper](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_35) "Generating Talking Face Landmarks from Speech."

You can find the project page [here](http://www2.ece.rochester.edu/projects/air/projects/talkingface.html).

An improved version of this project can be found [here](http://www2.ece.rochester.edu/projects/air/projects/3Dtalkingface.html).

## Installation

#### The project depends on the following Python packages:

* Keras --- 2.2.4
* Tensorflow --- 1.9.0
* Librosa --- 0.6.0
* opencv-python --- 3.3.0.10
* dlib --- 19.7.0
* tqdm 
* subprocess

#### It also depends on the following packages:
* ffmpeg --- 3.4.1
* OpenCV --- 3.3.0

The code has been tested on Ubuntu 16.04 and OS X Sierra and High Sierra. 

## Code Example

The generation code has the following arguments:

* -i --- Input speech file
    * See [this](http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load) link for supported audio formats.
* -m --- Input talking face landmarks model 
* -d --- Delay in terms of frames, where one frame is 40 ms
* -c --- Number of context frames
* -o --- Output path

You can run the following code to test the system:

```
python generate.py -i test_samples/test1.flac -m models/D40_C3.h5 -d 1 -c 3 -o results/D40_C3_test1
```
## Feature Extraction

You can run featureExtractor.py to extract features from videos directly. The arguments are as follows:

* -vp --- Input folder containing video files (if your video file types are different from .mpg or .mp4, please modify the script accordingly)
* -sp --- Path to shape_predictor_68_face_landmarks.dat. You can download this file [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat).
* -o --- Output file name

Usage: 

```
python featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file-name.hdf5
```

## Training

The training code has the following arguments:

* -i --- Input hdf5 file containing training data
* -u --- Number of hidden units
* -d --- Delay in terms of frames, where one frame is 40 ms
* -c --- Number of context frames
* -o --- Output folder path to save the model

Usage:

```
python train.py -i path-to-hdf5-train-file/ -u number-of-hidden-units -d number-of-delay-frames -c number-of-context-frames -o output-folder-to-save-model-file
```

## Citation

```
@inproceedings{eskimez2018generating,
  title={Generating talking face landmarks from speech},
  author={Eskimez, Sefik Emre and Maddox, Ross K and Xu, Chenliang and Duan, Zhiyao},
  booktitle={International Conference on Latent Variable Analysis and Signal Separation},
  pages={372--381},
  year={2018},
  organization={Springer}
}
```