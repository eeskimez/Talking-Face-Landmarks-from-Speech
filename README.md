# Generating Talking Face Landmarks

## Installation

#### The project depends on the following Python packages:

* Keras
* Tensorflow
* Librosa
* opencv-python
* dlib
* tqdm
* subprocess

#### It also depends on the following packages:
* ffmpeg
* OpenCV

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

