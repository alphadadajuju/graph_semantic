// ROS
#include <ros/ros.h>

// ROS messages
#include <std_msgs/Int16.h>
#include <std_msgs/Float64.h>
#include "std_msgs/String.h"

#include "graph_semantic/FusedPointcloud.h"

#include "graph_semantic/Cluster.h"
#include "graph_semantic/ClusterArray.h"
#include <std_msgs/Float32MultiArray.h>


// Headers C_plus_plus
#include <iomanip>
#include <iostream>
#include <string>
#include <csignal>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

// PCL CLustering
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

// OpenCV 
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;


// define cluster structure and its properties
struct ClusterStruct{
    float label, x, y, z, radius;
    vector<float> label_probs;

};

class ClustersPointcloud
{
private:
    bool visualize_clusters;
    cv::Mat label2bgr;                    // mapping from label to bgr

    int num_labels;                       // number of classes
    int min_cluster_size;
    int max_cluster_size;
    float cluster_tolerance;


    ros::NodeHandle nh;                 // node handler  
    ros::Subscriber sem_pc_msg_sub;     // dense semantic pointcloud (fused pointcloud)
    ros::Publisher clusters_msg_pub;    // cluster message

public:
    ClustersPointcloud();

    // CORE: clustering pointcloud by class AND distance
    void clusteringPointcloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr pc_xyzl, 
                              const vector<vector<float>> &label_probs, 
                              vector<ClusterStruct> &clusters);

    void sem_pc_callback(const graph_semantic::FusedPointcloud pc_fused_msg);
};

ClustersPointcloud::ClustersPointcloud()
{   
    // Make sure the following parameters are provided in the config file
    num_labels = -1;
    min_cluster_size = -1;
    max_cluster_size = -1 ;
    cluster_tolerance = -1 ;
    visualize_clusters = true;
    ros::param::get("graph_semantic/num_labels", num_labels);
    ros::param::get("graph_semantic/clusters_node/cluster_tolerance", cluster_tolerance); 
    ros::param::get("graph_semantic/clusters_node/min_cluster_size", min_cluster_size); 
    ros::param::get("graph_semantic/clusters_node/max_cluster_size", max_cluster_size); 
    ros::param::get("graph_semantic/publish_cluster_markers", visualize_clusters);

    // Read label-to-color mapping image
    string caffe_model_path;
    ros::param::get("graph_semantic/caffe_model_path", caffe_model_path);

    string label2bgr_filepath = caffe_model_path + "/sun_redux.png";
    label2bgr = cv::imread(label2bgr_filepath, CV_LOAD_IMAGE_COLOR);

    // Subscriber and Publisher
    sem_pc_msg_sub = nh.subscribe("graph_semantic/semantic_fused_pc", 10, &ClustersPointcloud::sem_pc_callback, this);
    clusters_msg_pub = nh.advertise<graph_semantic::ClusterArray>("graph_semantic/clusters", 10);
}

void ClustersPointcloud::sem_pc_callback(const graph_semantic::FusedPointcloud pc_fused_msg)
{   

    // conversion from ROS pointcloud2 msg to PCL pointcloud
    pcl::PCLPointCloud2 pcl_pc2; 
    pcl_conversions::toPCL(pc_fused_msg.labelled_pc, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZL>::Ptr pc_xyzl(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::fromPCLPointCloud2(pcl_pc2, *pc_xyzl);

    // Unpack multiarray prob_distribution into stl matrix
    vector<float> label_prob_init(num_labels, -500.0);
    vector<vector<float>> label_probs(pc_xyzl->points.size(), label_prob_init);

    for(int i = 0; i < label_probs.size(); i++){
        for(int j = 0; j < label_probs[0].size(); j++){
            label_probs[i][j] = pc_fused_msg.label_probs.data[j + pc_fused_msg.label_probs.layout.dim[1].stride*i];
        }
    }

    // Compute clusters of the pointcloud 
    vector<ClusterStruct> clusters;
    clusteringPointcloud(pc_xyzl, label_probs, clusters);

    // if visualize_clusters,display in Rviz
    if(visualize_clusters){

        graph_semantic::ClusterArray lc_msg;

        ROS_INFO_STREAM("There are " << clusters.size() << " clusters " << std::endl);
        
        for (int i = 0; i < clusters.size(); i++) {
            graph_semantic::Cluster c;

            c.label = clusters[i].label;
            c.x = clusters[i].x;
            c.y = clusters[i].y;
            c.z = clusters[i].z;
            c.radius = clusters[i].radius;
            copy(clusters[i].label_probs.begin(), clusters[i].label_probs.end(), back_inserter(c.label_probs));

            lc_msg.clusters.push_back(c);
        }

        clusters_msg_pub.publish(lc_msg);
    }

}

// CORE: clustering pointcloud by class AND distance
void ClustersPointcloud::clusteringPointcloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr pc_xyzl,
                                               const vector<vector<float> > &label_probs,
                                               vector<ClusterStruct> &clusters)
{   
    // Extracting cluster for each label except for wall, floor and ...
    for(int cl = 0; cl < num_labels; cl++){
        
        if(cl == 3 || cl == 4 || cl == 10){
            continue;
        }
        
        
        // cluster pointcloud by label
        vector<uint32_t> filtered_idx(0); //indices of filtered pointclouds 
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        for(uint32_t i = 0; i < pc_xyzl->points.size(); i++){
            if(pc_xyzl->points[i].label == cl){
                filtered_idx.push_back(i);
                pcl::PointXYZ p(pc_xyzl->points[i].x, pc_xyzl->points[i].y, pc_xyzl->points[i].z);
                pc_filtered-> points.push_back(p);
            }
        }

        if(pc_filtered->points.empty()) continue; 

        ROS_INFO_STREAM("semantic_cluster_node: Clustering label" << cl << ": " << std::endl);

        // further cluster same-label pointcloud by distance
        // Create KdTree instance for search method for cluster extraction
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(pc_filtered);

        vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec; 
        ec.setClusterTolerance(cluster_tolerance); // 2cm
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(pc_filtered);
        ec.extract(cluster_indices);

        // Find geometric center and radius of cluster
        for(vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end (); it++)
        {
            vector<float> cluster_label_probs(num_labels, 1.0f/(1.0f*num_labels)); // each cluster's confidence of having a particular label

            // find mean and make aggregate prob_dist
            pcl::PointXYZ p(0,0,0);
            for(vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++){
                p.x += pc_filtered->points[*pit].x ;
                p.y += pc_filtered->points[*pit].y ;
                p.z += pc_filtered->points[*pit].z ;

                 // Divide each element by the sum so that they add up to 1
                for (int cl = 0; cl < num_labels; cl++) {
                    cluster_label_probs[cl] += label_probs[filtered_idx[*pit]][cl];
                }

            }

            double prob_dist_sum = 0.0;
            for (int cl = 0; cl < num_labels; cl++){
                prob_dist_sum += cluster_label_probs[cl];
            }
            for (int cl = 0; cl < num_labels; cl++){
                cluster_label_probs[cl] /= prob_dist_sum;
            }

            // geometric center
            p.x /= (float) it->indices.size();
            p.y /= (float) it->indices.size();
            p.z /= (float) it->indices.size();

            // compute cluster radius (furthest point from the geometric center)
            float dist_max = 0 ;
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++){
                float delta_x = p.x - pc_filtered->points[*pit].x;
                float delta_y = p.y - pc_filtered->points[*pit].y;
                float delta_z = p.z - pc_filtered->points[*pit].z;

                float dist = delta_x * delta_x + delta_y*delta_y + delta_z*delta_z ;
                if (dist > dist_max){
                    dist_max = dist ;
                }
            }

            // wrap cluster to the defined cluster structure 
            ClusterStruct cluster;
            cluster.label = (uint8_t) cl;
            cluster.x = p.x;
            cluster.y = p.y;
            cluster.z = p.z;
            cluster.radius = std::sqrt(dist_max);
            copy(cluster_label_probs.begin(), cluster_label_probs.end(), back_inserter(cluster.label_probs));

            clusters.push_back(cluster);

        }

    }   
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "semantic_cluster_node");
    ClustersPointcloud cluster_pointcloud; 
    ros::spin();
    return 0;
}