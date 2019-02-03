# Written by S. Emre Eskimez, in 2017 - University of Rochester
# This script is written for extracting features from GRID dataset. 
# If you intend to use other videos with arbitrary length, you need to modify this script.
# Usage: python featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file-name.hdf5
# You can find shape_predictor_68_face_landmarks.dat online from various sources.
 
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from copy import deepcopy
import sys
import os
import dlib
import glob
# from skimage import io
import numpy as np
import h5py
import pylab
import librosa
import imageio
import utils
import argparse, fnmatch, shutil
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-vp", "--video-path", type=str, help="video folder path")
parser.add_argument("-sp", "--sp-path", type=str, help="shape_predictor_68_face_landmarks.dat path")
parser.add_argument("-o", "--output-path", type=str, help="Output file path")
args = parser.parse_args()

predictor_path = args.sp_path#'../data/shape_predictor_68_face_landmarks.dat'
video_folder_path = args.video_path
dataset_path = args.output_path

ms = np.load('mean_shape.npy') # Mean face shape, you can use any kind of face instead of mean face.
fnorm = utils.faceNormalizer()
ms = fnorm.alignEyePoints(np.reshape(ms, (1, 68, 2)))[0,:,:]

try:
    os.remove(dataset_path)
except:
    print ('Exception when deleting previous dataset...')

wsize = 0.04
hsize = 0.04

# These two vectors are for filling the empty cells with zeros for delta and double delta features
zeroVecD = np.zeros((1, 64))
zeroVecDD = np.zeros((2, 64))

dataHandler = h5py.File(dataset_path)

speechData = dataHandler.create_dataset('MelFeatures', (1, 75, 128), maxshape=(None, 75, 128))
lmarkData = dataHandler.create_dataset('flmark', (1, 75, 136), maxshape=(None, 75, 136))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

points_old = np.zeros((68, 2), dtype=np.float32)

fileCtr = 0

for root, dirnames, filenames in os.walk(video_folder_path):
    for filename in filenames:
        # You can add the file type of your videos here:
        if os.path.splitext(filename)[1] == '.mpg' or os.path.splitext(filename)[1] == '.mp4':
            f = os.path.join(root, filename)
            
            vid = imageio.get_reader(f,  'ffmpeg')
            point_seq = []
            img_seq = []

            for frm_cnt in tqdm(range(0, vid.get_length())):
                points = np.zeros((68, 2), dtype=np.float32)

                try:
                    img = vid.get_data(frm_cnt)
                except:
                    print('FRAME EXCEPTION!!')
                    continue

                dets = detector(img, 1)
                if len(dets) != 1:
                    print('FACE DETECTION FAILED!!')
                    continue

                for k, d in enumerate(dets):
                    shape = predictor(img, d)

                    for i in range(68):
                        points[i, 0] = shape.part(i).x
                        points[i, 1] = shape.part(i).y

                # points = np.reshape(points, (points.shape[0]*points.shape[1], ))
                point_seq.append(deepcopy(points))

            cmd = 'ffmpeg -y -i '+os.path.join(root, filename)+' -vn -acodec pcm_s16le -ac 1 -ar 44100 temp.wav'
            subprocess.call(cmd, shell=True) 

            y, sr = librosa.load('temp.wav', sr=44100)

            os.remove('temp.wav')
            frames = np.array(point_seq)
            fnorm = utils.faceNormalizer()
            aligned_frames = fnorm.alignEyePoints(frames)
            transferredFrames = fnorm.transferExpression(aligned_frames, ms)
            frames = fnorm.unitNorm(transferredFrames)

            if frames.shape[0] != 75:
                continue
        
            melFrames = np.transpose(utils.melSpectra(y, sr, wsize, hsize))
            melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
            melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
            melFeatures = np.concatenate((melDelta, melDDelta), axis=1)

            if melFeatures.shape[0] != 75:
                continue

            speechData[fileCtr, :, :] = melFeatures
            speechData.resize((speechData.shape[0]+1, 75, 128))

            lmarkData[fileCtr, :, :] = np.reshape(frames, (75, 136))
            lmarkData.resize((lmarkData.shape[0]+1, 75, 136))

            fileCtr += 1