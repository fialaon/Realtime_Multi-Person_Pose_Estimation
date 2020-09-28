# Realtime Multi-Person Pose Estimation
By [Zhe Cao](https://people.eecs.berkeley.edu/~zhecao/), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).

![Example output video](./testing/python/example_output.gif)

This is a fork of the original [CVPR'17 implementation of Openpose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (without hand and facial landmarks) adapted to the [3D motion-force estimator](https://www.di.ens.fr/willow/research/motionforcesfromvideo/).

This version is different from the official releases in the following aspects:
- A different I/O interface:
  - The input is required to be an *image folder* (explained [here](https://github.com/zongmianli/contact-recognizer/blob/public/doc/create_data.md)) containing a mixture of still images and video frame images. 
  - The estimated 2D poses are output in the data structure required by the [contact recognizer](https://github.com/zongmianli/contact-recognizer) and the [3D motion-force estimator](https://www.di.ens.fr/willow/research/motionforcesfromvideo/).  
- Adaptation to the scenario of *object manipulation in instructional videos*.
  The following post-processing steps are appended to the testing module of Openpose:
  - We assume that there is at most one person in each input image / video frame.
    When multiple human instances are present, only the one detected with the highest score are preserved — the others are ignored.
  - Due to the heavy occlusion during person-object interaction, the predicted Fart Affinity Fields (PAFs) may not be correct. 
    As a result, some joints (often hands, ankles) may be missing or mis-detected, e.g. associated to another people in the background.
    To address this problem we have made changes in the bottom-up parsing step in the original implementation.

## Installation

```terminal
git clone https://github.com/zongmianli/Realtime_Multi-Person_Pose_Estimation ~/Openpose-video
```

See the original [repo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) for more information if needed.

## Testing

Assume that we have a mixture of still images and videos to test.

1. Follow Step 1-3 in this [doc](https://github.com/zongmianli/contact-recognizer/blob/public/doc/create_data.md) to create a new image folder with a arbitrary name, for example, `sample_imagefolder`, and put the folder to the path `~/Openpose-video/testing/sample_imagefolder`.

2. Go to `~/Openpose-video/testing/python/` and run:
      ```terminal
   img_dir=~/Openpose-video/testing/sample_imagefolder
   vis_dir=~/Openpose-video/testing/sample_imagefolder_vis
   save_path=${vis_dir}/Openpose-video.pkl
   python run_imagefolder.py ${img_dir} ${vis_dir} ${save_path} --save-after-each-iteration
   ```

The estimated 2d poses and the corresponding visualization images will be saved in `${save_path}` and `${vis_dir}`, respectively.

