# Openpose-MFV: Human 2D Pose Estimation for MFV

Openpose-MFV is a fork of the original [CVPR'17 implementation of Openpose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (without hand and facial landmarks) adapted for the 3D motion and force estimator, [MFV](https://www.di.ens.fr/willow/research/motionforcesfromvideo/).

![Example output video](https://github.com/zongmianli/Realtime_Multi-Person_Pose_Estimation/blob/master/testing/python/example_output.gif)


Comparing with the original version, this new version has the following major changes in the testing code:
- **An easy interface** for testing on multiple images and/or videos.
- **A revised bottom-up parsing step** which is adapted for object manipulation scenarios in instructional videos:
   - We assume that there is **a single human subject** in the input video/image. When multiple people are detected, only the most confident human body will be perserved — all the other human instances will be ignored.
   - Due to occlusion with the manipulated object, the predicted Fart Affinity Fields (PAFs) may not be correct. As a result, some joints (for example, hand, ankle) of the detected human subject will be missing or associated to wrong people (who are ignored eventually). 
   To deal with this case, we always associate the most confident joint detection to the output human 2D skeleton.

## Testing on still images
The following scripts detect images (with extension `.jpg` or `.png`) under the `input_dir` directory and run Openpose-MFV on the detected images.
Visualization images and estimated 2D joint locations are saved in `vis_dir` and `save_path`, respectively.
```terminal
$ cd testing/python
$ input_dir=path/to/target_folder
$ vis_dir=${input_dir}/Openpose-MFV
$ save_path=${input_dir}/Openpose-MFV/person_2d_joints.pkl
$ python main.py ${input_dir} ${vis_dir} ${save_path}
```

## Testing on videos
Suppose we have a folder containing videos and we wish to do a forward pass on all the videos in a frame-by-frame manner to obtain an estimate of the 2D trajectory of human joints.

### Convert videos to image sequences
First, convert the input videos to individual frames and save the frame images under folders with video names. Taking the Handtool dataset as an example, we expect the following folder sturcture:
```terminal
$ image_folder=path/to/Handtool-dataset/frames
$ ls ${image_folder}
barbell_1	hammer_1	scythe_1	spade_1
barbell_2	hammer_2	scythe_2	spade_2
barbell_3	hammer_3	scythe_3	spade_3
barbell_4	hammer_4	scythe_4	spade_4
barbell_5	hammer_5	scythe_5	spade_5
$ ls ${videos_dir}/barbell_1/
0001.png	0002.png	0003.png	...
```
Namely, we expect an `image_folder` containing several subfolders of frame images. 
Under each subfolder, e.g. barbell_1, are saved the frame images corresponding to an input video.

### Generate `data_info.pkl` for the `image_folder`
```terminal
$ cd testing/python
$ python helpers/create_data_info.py ${image_folder} --image_types="png,jpg" --save_info
```
### Run Openpose-MFV on the `image_folder`
```terminal
$ cd testing/python
$ vis_folder=path/to/Handtool-dataset/Openpose-MFV
$ save_path=${vis_folder}/Openpose-MFV.pkl
$ python run_imagefolder.py ${image_folder} ${vis_folder} ${save_path} --save-after-each-iteration
```

## Testing on a mixture of images and videos
To run Openpose-MFV on a mixture of still images and video frames, it is sufficient put video frames as subfolders in `image_folder` and still images directly under `image_folder`.
Then follow exactly the same steps described in "Testing on videos".

This feature is handy for collecting data (both still images and video clips) for training the contact recognizer for MFV.
