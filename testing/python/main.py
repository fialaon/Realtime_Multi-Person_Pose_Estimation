import cv2 as cv
import numpy as np
import numpy.linalg as LA
import scipy
from scipy.ndimage.filters import gaussian_filter
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
import pickle as pk
from glob import glob
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile
import argparse

def main(image_paths, vis_dir, save_path=None):
    '''
    Run Openpose on a set of images.
    -
    image_paths: a list of image paths, e.g. ['path/to/image1', 'path/to/image2']
    vis_dir: folder path for saving output images
    save_path: the estimated human 2D poses will be saved to file if a valid save_path is provided
    '''

    num_images = len(image_paths)
    facial_landmarks = [0, 14, 15, 16, 17]

    # ------------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------------

    param, model = config_reader()
    param['scale_search'] = [0.6]#, 0.8, 1.0, 1.2]
    if param['use_gpu']:
        caffe.set_mode_gpu()
        caffe.set_device(param['GPUdeviceNumber']) # set to your device!
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

    # Iterate over input images
    joints_2d = np.zeros((num_images, 18, 3))
    for img_id in range(num_images):
        image_path = image_paths[img_id]
        print("Processing {} ...".format(image_path))
        oriImg = cv.imread(image_path) # B,G,R order

        # ------------------------------------------------------------
        # Resize image to multiple scales and do forward pass
        # ------------------------------------------------------------

        multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])

            net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            #net.forward() # dry run
            net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
            start_time = time.time()
            output_blobs = net.forward()
            print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

            # Extract outputs, resize, and remove padding
            heatmap = np.transpose(np.squeeze(net.blobs[list(output_blobs.keys())[1]].data), (1,2,0)) # output 1 is heatmaps
            heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            paf = np.transpose(np.squeeze(net.blobs[list(output_blobs.keys())[0]].data), (1,2,0)) # output 0 is PAFs
            paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        # ------------------------------------------------------------
        # Compute all peaks with score. allocate an id to each peak
        # ------------------------------------------------------------

        all_peaks = []
        peak_counter = 0
        for part in range(18):
            x_list = []
            y_list = []
            map_ori = heatmap_avg[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]

            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
            peaks = [o for o in zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])] # note reverse
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # ------------------------------------------------------------
        # Define links
        # ------------------------------------------------------------

        # Find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                   [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                   [1,16], [16,18], [3,17], [6,18]]
        # The middle joints heatmap correpondence
        mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
                  [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
                  [55,56], [37,38], [45,46]]

        # ------------------------------------------------------------
        # Compute connections
        # ------------------------------------------------------------

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0]-1]
            candB = all_peaks[limbSeq[k][1]-1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if(nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        if candB[j][:2]==candA[i][:2]:
                            vec = np.array([1.,1.])
                        else:
                            vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        vec = np.divide(vec, norm)

                        startend = [o for o in zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))]

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0,5))
                for c in range(len(connection_candidate)):
                    i,j,s = connection_candidate[c][0:3]
                    if(i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if(len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # ------------------------------------------------------------
        # compute subsets
        # ------------------------------------------------------------

        # Last number in each row is the total parts number of that person
        # Second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print("found = 2")
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # Create a new subset if partA is not found in the subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        # ------------------------------------------------------------
        # Keep the most confident subset
        # ------------------------------------------------------------

        # Initialize the person with no joint and zero confidence
        person = -1*np.ones((20)) # no peaks
        person[-1] = 0. # no detected joints
        person[-2] = 0. # zero score for that person
        c_max = 0.
        if len(subset)>0:
            for i in range(len(subset)):
                if subset[i][-2]>c_max:
                    c_max = subset[i][-2]
                    person = subset[i]
        else:
            continue
        # Assign the most confident joint peak to missing joints in person
        for i in range(18):
            if person[i]== -1 and len(all_peaks[i])>0:
                # seach the peak with highest score
                joint_peaks = all_peaks[i]
                max_score = 0.
                pid = -1
                for k in range(len(joint_peaks)):
                    if joint_peaks[k][2]>max_score:
                        max_score = joint_peaks[k][2]
                        pid = joint_peaks[k][3]
                person[i] = pid

        # Estimate the position of the person's chest
        chest_joint_positions = []
        chest_location = None
        for i in [1,2,5]: # neck, l/r shoulder
            pid = person[i].astype(int)
            if pid >= 0:
                for j in range(len(all_peaks[i])):
                    if all_peaks[i][j][3] == pid:
                        joint_position = np.array(all_peaks[i][j][:2]) # 1d array
                        break
                chest_joint_positions.append(joint_position)
        if len(chest_joint_positions) > 0:
            chest_joint_positions = np.array(chest_joint_positions)
            chest_location = np.mean(chest_joint_positions, axis=0)

        # Remove pid from person if the peak is ...
        removed_joints = []
        for i in range(18):
            pid = person[i].astype(int)
            if pid >= 0:
                for j in range(len(all_peaks[i])):
                    if all_peaks[i][j][3] == pid:
                        joint_position = np.array(all_peaks[i][j][0:3]) # 1d array
                        break

                # For facial landmarks, remove the landmark if it is very far from the person's chest
                remove_joint = False
                if i in facial_landmarks and chest_location is not None:
                    dist_from_the_chest = LA.norm(joint_position[:2] - chest_location)
                    if dist_from_the_chest > 0.18*oriImg.shape[0]:
                        remove_joint = True

                if remove_joint:
                    person[i] = -1 # set the pid back to -1
                    removed_joints.append(i)
                else:
                    joints_2d[img_id][i] = joint_position

        # ------------------------------------------------------------
        # Draw estimated joints on input images
        # ------------------------------------------------------------
        colors_rgb = [[255,153,51], # nose: orange
					  [255,153,51], # neck: orange
					  [127,255,0],[127,255,0],[127,255,0], # right arm: green
					  [255,215,0],[255,215,0],[255,215,0], # left arm: yellow
					  [0,255,255],[0,255,255],[0,255,255], # right leg: cyan
					  [255,0,255],[255,0,255],[255,0,255], # left leg: pink
					  [127,255,0], # right eye: green
					  [255,215,0], # left eye: yellow
					  [0,255,255], # right ear: cyan
					  [255,0,255]]

        colors = [[c[2], c[1], c[0]] for c in colors_rgb] # convert to BGR
        cmap = matplotlib.cm.get_cmap('hsv')
        canvas = cv.imread(image_path) # B,G,R order

        for i in range(18):
            rgba = np.array(cmap(1 - i/18. - 1./36))
            rgba[0:3] *= 255
            pid = person[i].astype(int)
            for j in range(len(all_peaks[i])):
                if all_peaks[i][j][3] == pid:
                    # Draw smaller circles for facial landmarks than other body joints
                    csize = 3 if i in facial_landmarks else 5
                    cv.circle(canvas, all_peaks[i][j][0:2], csize, colors[i], thickness=-1)

        # ------------------------------------------------------------
        # Draw links (or "limbs")
        # ------------------------------------------------------------
        stickwidth = 2

        for i in range(17):
            index = person[np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            # Draw them limbs in grey color
            grey_color = [255, 255, 255]
            cv.fillConvexPoly(cur_canvas, polygon, grey_color)
            canvas = cv.addWeighted(canvas, 0.6, cur_canvas, 0.4, 0)

        # ------------------------------------------------------------
        # Save the image to file
        # ------------------------------------------------------------
        if not exists(vis_dir):
            makedirs(vis_dir)

        vis_path = join(vis_dir, basename(image_path))
        cv.imwrite(vis_path, canvas)

    # ------------------------------------------------------------
    # Optionally, save joint locations to file
    # ------------------------------------------------------------
    if save_path is not None:
        data_dict = {
            "joint_2d_positions": joints_2d,
            "image_names": [basename(image_paths[i]) for i in range(num_images)]
        }
        with open(save_path, 'w') as f:
            pk.dump(data_dict, f)

    return joints_2d

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run Openpose on all images within a folder.')
    parser.add_argument(
        "input_dir", help="Path to a folder containing images.")
    parser.add_argument(
        "vis_dir", help="Path to another folder for saving output visualization images")
    parser.add_argument(
        "save_path", help="Path for saving output joint locations.")

    args = parser.parse_args()
    input_dir = args.input_dir
    vis_dir = args.vis_dir
    save_path = args.save_path

    # Retrieve image paths from the input image folder
    image_extensions = ("jpg", "png")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(sorted(glob(join(input_dir, "*.{0:s}".format(ext)))))

    main(image_paths, vis_dir, save_path)
