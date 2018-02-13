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
import itertools

import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
#from rp_semantic.msg import Frame
from graph_semantic.msg import Cluster
from graph_semantic.msg import ClusterArray
#from graph_semantic.msg import BoWP

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class ClusterVis:
    def __init__(self):
        print ('Program begins:')
        self.num_labels = rospy.get_param("graph_semantic/num_labels", 37)

        # define class variables
        self.label_cluster_ready = False
        self.wait_for_message = True

        model_base_path = rospy.get_param("graph_semantic/caffe_model_path", '')
        colours_path = model_base_path + 'sun_redux.png'
        colours_img = cv2.imread(colours_path)
        self.label_colours = colours_img.astype((np.uint8))

        self.semantic_classes = ['bed', 'books', 'tv', 'ceiling', 'floor', 'seat',
                                 'table', 'door', 'window', 'poster', 'wall', 'closet',
                                 'shelves', 'toilet', 'sink', 'pillow', 'refrigerator', 'objects']

        # subscriber
        self.label_cluster_sub = rospy.Subscriber('/graph_semantic/clusters', ClusterArray, self.cluster_array_callback, queue_size=1, buff_size=2 ** 24) # change topic's name accordingly

        # publisher
        self.sphere_pub = rospy.Publisher('/graph_semantic/marker_cluster', MarkerArray, queue_size=10) # change topic's name accordingly


    def cluster_array_callback(self, label_cluster_msg):
        self.cluster_array = label_cluster_msg

        num_cluster = np.shape(self.cluster_array.clusters)[0]
        rospy.loginfo(str(num_cluster) + ' clusters in this frame')

        # Create DELETEALL marker msg
        m_delete  = Marker()
        m_delete.header.frame_id = "/map"
        m_delete.action = m_delete.DELETEALL

        # Put it into a markerarray to reuse the topic
        m_deletearray = MarkerArray()
        m_deletearray.markers.append(m_delete)
        self.sphere_pub.publish(m_deletearray)

        self.markerArray = MarkerArray()
        for i in range(0, num_cluster):
            cluster_label = int(self.cluster_array.clusters[i].label)
            if cluster_label >= self.num_labels:
                continue

            marker = self.cluster2marker(self.cluster_array.clusters[i])
            marker.id = i
            self.markerArray.markers.append(marker)

        rospy.loginfo('number of markers drawn:' + str(np.shape(self.markerArray.markers)[0]))
        self.sphere_pub.publish(self.markerArray)


    def cluster_array_debug_callback(self, label_cluster_msg):
        self.cluster_array = label_cluster_msg

        num_cluster = np.shape(self.cluster_array.clusters)[0]
        rospy.loginfo(str(num_cluster) + ' clusters in this frame')

        for i in range(0, num_cluster):
            # Create DELETEALL marker msg
            m_delete  = Marker()
            m_delete.header.frame_id = "/map"
            m_delete.action = m_delete.DELETEALL

            # Put it into a markerarray to reuse the topic
            m_deletearray = MarkerArray()
            m_deletearray.markers.append(m_delete)
            self.sphere_pub.publish(m_deletearray)

            cluster_label = int(self.cluster_array.clusters[i].label)
            if cluster_label >= self.num_labels:
                continue

            self.markerArray = MarkerArray()
            marker = self.cluster2marker(self.cluster_array.clusters[i])
            marker.id = i
            self.markerArray.markers.append(marker)

            rospy.loginfo('number of markers drawn:' + str(np.shape(self.markerArray.markers)[0]))
            self.sphere_pub.publish(self.markerArray)

            # Print class label probability
            print('Cluster ' + str(i) + " " + self.semantic_classes[int(np.argmax(self.cluster_array.clusters[i].label_probs))])
            for j in range(len(self.semantic_classes)):
                print(self.semantic_classes[j] + '= ' + str(round(self.cluster_array.clusters[i].label_probs[j], 3)))

            cluster_conf = 0.0
            for j in range(self.num_labels):
                cluster_conf += -np.log(self.cluster_array.clusters[i].label_probs[j])

            print('Cluster confidence: ' + str(round(cluster_conf,4)))

            raw_input("Press Enter to continue...")

    def cluster2marker(self, cluster):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        marker.color.a = 0.6
        marker.pose.orientation.w = 1.0

        marker.pose.position.x = cluster.x
        marker.pose.position.y = cluster.y
        marker.pose.position.z = cluster.z
        marker.scale.x = cluster.radius * 2.0
        marker.scale.y = cluster.radius * 2.0
        marker.scale.z = cluster.radius * 2.0

        marker.color.r = round(self.label_colours[0][cluster.label][2] / 255.0, 1)
        marker.color.g = round(self.label_colours[0][cluster.label][1] / 255.0, 1)
        marker.color.b = round(self.label_colours[0][cluster.label][0] / 255.0, 1)

        return marker

if __name__ == '__main__':

    rospy.init_node( 'cluster_visualization', log_level=rospy.INFO)

    cluster_v_node = ClusterVis()
    rospy.spin()