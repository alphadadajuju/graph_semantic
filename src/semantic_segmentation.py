#!/usr/bin/env python
# ROS imports
import roslib; roslib.load_manifest('graph_semantic')
import rospy
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
from cv_bridge import CvBridge, CvBridgeError
import sys
import time

"""
LOGLEVEL OF 2 SUPPRESSES OUTPUT OF CAFFE!!!!!!!!!! Put it to 0 to turn it back on
"""
#os.environ['GLOG_minloglevel'] = '2'
import caffe


import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
#from rp_semantic.msg import Frame
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

from graph_semantic.srv import *

class SegnetSemantic:

    def __init__(self):
        # class variable initialization
        self.rgb_frame = None
        self.height = rospy.get_param("graph_semantic/sensor_image_height", 480)
        self.width = rospy.get_param("graph_semantic/sensor_image_width", 640)
        self.num_labels = rospy.get_param("graph_semantic/num_labels", 37)

        self.publish_rgb_frame = rospy.get_param("graph_semantic/publish_rgb_frame", False)
        self.publish_rgb_labels = rospy.get_param("graph_semantic/publish_rgb_labels", False)

        # Flags and variables for exchange of data between service callback and main control loop (faster if main loop segments!)
        self.wait_for_segnet = False
        self.obtain_rgb = False
        self.class_response = Float32MultiArray()
        self.semantic_label_message = Image()


        # SEGMENTATION parameters

        # Caffe configuration
        caffe.set_mode_gpu()
        caffe.set_device(0)

        caffe_root = rospy.get_param("graph_semantic/caffe_root_path", '/home/alpha/github/caffe-segnet-cudnn5/')
        model_base_path = rospy.get_param("graph_semantic/caffe_model_path", "/home/alpha/catkin_ws/src/segnet_program/src/")

        sys.path.append('/usr/local/lib/python2.7/site-packages')
        sys.path.insert(0, caffe_root + 'python')

        model_path = model_base_path + 'segnet_sun.prototxt'
        weights_path = model_base_path + 'segnet_sun.caffemodel'
        colours_path = model_base_path + 'sun_redux.png'

        # Load NET
        self.net = caffe.Net(model_path, weights_path, caffe.TEST)
        self.input_shape = self.net.blobs['data'].data.shape  # 1 x 3 x 224 x 224
        self.output_shape = self.net.blobs['prob'].data.shape  # 1 x 1 x 224 x 224

        # Load label color image
        colours_img = cv2.imread(colours_path)
        if colours_img is None:
            rospy.logerr("COULDN'T LOAD COLOURS IMAGE")
            exit()
        self.label_colours = colours_img.astype(np.uint8)
        self.bridge = CvBridge()  # for decoding sensor_msgs Image data[]

        # ROS services and publishers
        self.s = rospy.Service('rgb_to_label_prob', RGB2LabelProb, self.segmentation_callback)
        self.semantic_rgb_frame_pub = rospy.Publisher('/graph_semantic/semantic_rgb_frame', Image, queue_size=1)
        self.rgb_frame_pub = rospy.Publisher('/graph_semantic/rgb_frame', Image, queue_size=1)

        print ('Semantic initialized')


    def reduceSegNetclasses(self, probs):
        """
        Reduces SegNet probability output from 37 classes to self.num_labels
        """

        # Map going from 37 to 19 classes SemanticPRv1 mapping
        reduction_map = [10, 4, 11, 0, 5, 5, 6, 7, 8, 12, 9, 6, 8, 6, 12, 8, 11, 15, 17, 4, 17,
                         3, 1, 16, 2, 9, 17, 17, 17, 9, 17, 6, 13, 14, 17, 14, 17]

        print probs.shape[2]
        print len(reduction_map)
        if len(reduction_map) != probs.shape[2]:
            rospy.logerr("provided reduction mapping is inconsistent with input num_labels")

        print self.num_labels
        if np.amax(reduction_map) != self.num_labels-1:
            rospy.logerr("num_labels in config is not consistent with provided mapping")

        #TODO optimize this, method is not quite efficient (nothing too bad)
        probs_out = np.zeros((probs.shape[0], probs.shape[1], self.num_labels))
        for i in range(0,probs.shape[2]):
            probs_out[:,:,reduction_map[i]] = np.maximum(probs_out[:,:,reduction_map[i]], probs[:,:,i])

        return probs_out


    def createMultiArrayMessage(self, img):
        """
        img: numpy ndarray or matrix with dimensions as (row, col, depth) or (height, width, label)

        """

        response = Float32MultiArray()

        # initialize based on multiarrayLayout definition (see online)
        response.layout.data_offset = 0

        response.layout.dim.extend([MultiArrayDimension() for i in range(0,3)])
        response.layout.dim[0].label = "height"
        response.layout.dim[0].size = img.shape[0]
        response.layout.dim[0].stride = np.prod(img.shape[0:3])
        response.layout.dim[1].label = "width"
        response.layout.dim[1].size = img.shape[1]
        response.layout.dim[1].stride = np.prod(img.shape[1:3])
        response.layout.dim[2].label = "class"
        response.layout.dim[2].size = img.shape[2]
        response.layout.dim[2].stride = img.shape[2]

        #Fill data
        response.data = np.reshape(img, img.size, 'C')

        return response



    def segmentation_callback(self, req):
        rospy.loginfo('semantic_segmentation: receive a request!')
        
        if self.wait_for_segnet is False: # if main control loop not busy
            try:
                self.rgb_frame = self.bridge.imgmsg_to_cv2(req.rgb_image, "bgr8")
            except CvBridgeError, e:
                    rospy.logerr("segmentation_callback: Conversion from rosmsg to cv mat failed")
                    exit()

            self.obtain_rgb = self.wait_for_segnet = True
            while self.wait_for_segnet is True: # main control loop still processing segnet and wrapping multiarray
                rospy.sleep(0.005)
                pass

            if self.publish_rgb_frame:
                self.rgb_frame_pub.publish(req.rgb_image)
            if self.publish_rgb_labels:
                self.semantic_rgb_frame_pub.publish(self.semantic_label_message)

            return RGB2LabelProbResponse(self.class_response)


    def controlLoopIteration(self):
            if self.obtain_rgb is False:
                return

            self.rgb_frame = cv2.resize(self.rgb_frame, (self.input_shape[3], self.input_shape[2]))
            b,g,r = cv2.split(self.rgb_frame)       # get b,g,r
            self.rgb_frame = cv2.merge([r,g,b])     # switch it to rgb

            input_image = np.asarray([self.rgb_frame.transpose((2, 0, 1))])
            while input_image.shape[3] != self.input_shape[3]:
                input_image = self.rgb_frame.transpose((2, 0, 1))


            # SEGMENTATION
            start = time.time()
            out = self.net.forward_all(data=input_image)
            end = time.time()

            rospy.loginfo('semantic_segmentation: Executed SegNet in %.6s ms.', str((end - start) * 1000))

            ## MULTIARRAY
            start = time.time()

            # Extract data from network
            segnet_prob_out = self.net.blobs['prob'].data.squeeze()
            segnet_prob_out = np.transpose(segnet_prob_out, axes=(1,2,0))

            # MINMAX normalization
            #pout_min, pout_max = np.amin(segnet_prob_out), np.amax(segnet_prob_out)
            #segnet_prob_out = (segnet_prob_out - pout_min)/(pout_max - pout_min)

            # Reduce number of classes
            segnet_prob_out = self.reduceSegNetclasses(segnet_prob_out)

            # Reshape segmented images to match camera original resolution
            segnet_reshaped_prob_out = np.zeros((self.height, self.width, self.num_labels))
            for cl in range(0, self.num_labels):
                segnet_reshaped_prob_out[:,:,cl] = cv2.resize(segnet_prob_out[:,:,cl], (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            # Reshape and make msg
            self.class_response = self.createMultiArrayMessage(segnet_reshaped_prob_out)

            end = time.time()
            rospy.loginfo('semantic_segmentation: Executed multiarray in %.6s ms.', str((end - start) * 1000))

            # Create RGB labels image
            if self.publish_rgb_labels:
                segmentation_ind = np.squeeze(self.net.blobs['argmax'].data) # squeeze removes dim = 1 (1x3x24 => 3x24)
                segmentation_ind_3ch = np.resize(segmentation_ind, (3, self.input_shape[2], self.input_shape[3]))
                segmentation_ind_3ch = segmentation_ind_3ch.transpose(1, 2, 0).astype(np.uint8)
                segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
                cv2.LUT(segmentation_ind_3ch, self.label_colours, segmentation_rgb)
                segmentation_rgb = segmentation_rgb.astype(np.uint8)
                self.semantic_label_message = self.bridge.cv2_to_imgmsg(np.uint8(segmentation_rgb), "bgr8")

            self.wait_for_segnet = self.obtain_rgb = False


if __name__ == '__main__':

    caffe.set_mode_gpu()
    caffe.set_device(0) # set gpu device

    rospy.init_node( 'semanticRGB_server', log_level=rospy.INFO)

    seg_node = SegnetSemantic()

    while not rospy.is_shutdown():
        seg_node.controlLoopIteration()

    rospy.spin()
