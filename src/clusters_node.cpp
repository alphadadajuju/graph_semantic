// ROS
#include <ros/ros.h>

// ROS messages
#include <std_msgs/Int16.h>
#include <std_msgs/Float64.h>
#include "std_msgs/String.h"

#include "rp_semantic/FusedPointcloud.h"

#include "rp_semantic/Cluster.h"
#include "rp_semantic/ClusterArray.h"
#include "rp_semantic/BoWP.h"
#include "rp_semantic/BoWPDescriptors.h"

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
#include <std_msgs/Float32MultiArray.h>


using namespace std;

struct ClusterStruct{
    float label;
    float x;
    float y;
    float z;
    float radius;
    vector<float> label_probs;
};


class ClustersPointClouds{

private:
    bool visualize_clusters;
    cv::Mat label2bgr;

    int num_labels ;
    int min_cluster_size ;
    int max_cluster_size ;
    float cluster_tolerance ;
    ros::NodeHandle cl_handle;            // node handler  
    ros::Subscriber segnet_msg_sub ;      // SegNet Message Subscriber
    ros::Subscriber pointCloud2_msg_sub ; // PointCloud2 message Subscriber
    ros::Publisher clusters_msg_pub ;    // Clusters node Message Publisher
    ros::Publisher descriptors_msg_pub ;  // Clusters node descriptors message

public:
    ClustersPointClouds();

    void clusteringPointcloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud ,
                              const vector<vector<float> > &label_probs,
                              vector<ClusterStruct> &clusters);

    void computeBoW( const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,  std::vector<float> &boW_descriptor );
    void computeBoWP(const std::vector<ClusterStruct> clusters, std::vector<float> &boWP_descriptor );

    void bowCallback(const rp_semantic::FusedPointcloud pc_fused_msg);
};



ClustersPointClouds::ClustersPointClouds(){

    // Initialization of variables
    num_labels = -1;
    ros::param::get("rp_semantic/num_labels", num_labels) ; // we can optimize it later
    
    min_cluster_size = -1 ;
    max_cluster_size = -1 ;
    cluster_tolerance = -1 ;
    ros::param::get("rp_semantic/clusters_node/cluster_tolerance", cluster_tolerance) ; // we can optimize it later
    ros::param::get("rp_semantic/clusters_node/min_cluster_size", min_cluster_size) ; // we can optimize it later
    ros::param::get("rp_semantic/clusters_node/max_cluster_size", max_cluster_size) ; // we can optimize it later

    visualize_clusters = true;
    ros::param::get("rp_semantic/publish_cluster_markers", visualize_clusters) ; // we can optimize it later

    // Read color to labels image
    string caffe_model_path;
    ros::param::get("rp_semantic/caffe_model_path", caffe_model_path);

    string label2bgr_filepath = caffe_model_path + "/sun_redux.png";
    label2bgr = cv::imread(label2bgr_filepath, CV_LOAD_IMAGE_COLOR);


    // Subscribers  // Topic subscribe to : rp_semantic/labels_pointcloud
    pointCloud2_msg_sub = cl_handle.subscribe("rp_semantic/semantic_fused_pc", 10, &ClustersPointClouds::bowCallback , this ); // Subscriber
    // Publishers
    descriptors_msg_pub = cl_handle.advertise<rp_semantic::BoWPDescriptors>("rp_semantic/place_descriptors", 10); // Publisher
    clusters_msg_pub = cl_handle.advertise<rp_semantic::ClusterArray>("rp_semantic/clusters", 10); // Publisher

}


void ClustersPointClouds::bowCallback(const rp_semantic::FusedPointcloud pc_fused_msg ){

    // Unpack sensor_msgs/Pointcloud2 in pc_msg.raw_pointcloud into a pointcloud
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL( pc_fused_msg.labelled_pc , pcl_pc2 );
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_in( new pcl::PointCloud<pcl::PointXYZL> );  // Pointcloud in of type PointXYZRGB
    pcl::fromPCLPointCloud2(pcl_pc2,*cloud_in);

    // Unpack multiarray into stl matrix
    vector<float> label_prob_init(num_labels, -500.0);
    vector<vector<float> > label_probs(cloud_in->points.size(), label_prob_init);

    for (int i = 0; i < label_probs.size(); ++i) {
        for (int j = 0; j < label_probs[0].size(); ++j) {
            label_probs[i][j] = pc_fused_msg.label_probs.data[j + pc_fused_msg.label_probs.layout.dim[1].stride*i];
        }
    }

    ROS_INFO_STREAM("Computing descriptors...");

    // Compute boW of the clusters  
    std::vector<float>  boW( num_labels ,0 ) ;
    computeBoW( cloud_in, boW );

    // Compute clusters of the Point cloud 
    std::vector<ClusterStruct>  clusters;
    clusteringPointcloud( cloud_in, label_probs, clusters);

    // Compute boWP of the cluster
    std::vector<float>  boWp(num_labels*num_labels, 0) ;
    computeBoWP( clusters, boWp );

    ROS_INFO_STREAM("Computed descriptors, publishing...");

    // Publishing BoW and boWP 
    rp_semantic::BoWPDescriptors msg_descriptors;

    // Filling the boW and boWP descripto message
    std::copy( boW.begin(), boW.end() , back_inserter(msg_descriptors.bow) ) ;
    std::copy( boWp.begin(), boWp.end() , back_inserter(msg_descriptors.bowp) ) ;

    descriptors_msg_pub.publish( msg_descriptors );

    //CLUSTER VISUALIZATION IS DISABLED UNTIL we adapt the cluster struct to the Cluster Ros msg in the copy function
    if(visualize_clusters){
        rp_semantic::ClusterArray lc_msg;

        for (int i = 0; i < clusters.size(); ++i) {
            rp_semantic::Cluster c;

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

} // end of bowcallback function


void ClustersPointClouds::clusteringPointcloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                                               const vector<vector<float> > &label_probs,
                                               vector<ClusterStruct> &clusters )
{
    // Extracting cluster for each labels
    for(int cl = 0 ; cl < num_labels ; cl ++ ){
        if(cl == 3 || cl == 4 || cl == 10){
            continue;
        }

        // Create the filtering object
        vector<uint32_t> filtered_idx(0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_labels(new pcl::PointCloud<pcl::PointXYZ> ); // Filtered point cloud of type pcl::PointXYZL


        //TODO when we add a point to the new cloud, erase it from the old to not rescan it over and over
        //TODO BUT! this is trickier than it sounds given the index referencing that we do :(
        for (uint32_t i = 0; i < cloud->points.size(); i++)
        {
            if(cloud->points[i].label == cl){
                filtered_idx.push_back(i);

                pcl::PointXYZ p(cloud->points[i].x , cloud->points[i].y, cloud->points[i].z );
                cloud_filtered_labels->points.push_back(p) ;
            }
        }

        if( cloud_filtered_labels->points.empty()) continue ;


        ROS_DEBUG_STREAM("Clustering label  " << cl << ": " << std::endl);
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (cloud_filtered_labels);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance (cluster_tolerance); // 2cm
        ec.setMinClusterSize (min_cluster_size);
        ec.setMaxClusterSize (max_cluster_size);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_filtered_labels);
        ec.extract (cluster_indices);


        //Find geometric center and radius of cluster
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
            vector<float> cluster_label_probs(num_labels, 1.0f/(1.0f*num_labels));// 0.0f);

            //find mean and make aggregate prob_dist
            pcl::PointXYZ p(0,0,0);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
                p.x += cloud_filtered_labels->points[*pit].x ;
                p.y += cloud_filtered_labels->points[*pit].y ;
                p.z += cloud_filtered_labels->points[*pit].z ;

                // Divide each element by the sum so that they add up to 1
                for (int cl = 0; cl < num_labels; ++cl) {
                    cluster_label_probs[cl] += label_probs[filtered_idx[*pit]][cl];
                }
            }

            double prob_dist_sum = 0.0;
            for (int cl = 0; cl < num_labels; ++cl) {
                prob_dist_sum += cluster_label_probs[cl];
            }
            for (int cl = 0; cl < num_labels; ++cl) {
                cluster_label_probs[cl] /= prob_dist_sum;
            }

            p.x /= (float) it->indices.size();
            p.y /= (float) it->indices.size();
            p.z /= (float) it->indices.size();

            //compute dist
            float dist_max = 0 ;
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
                float delta_x = p.x - cloud_filtered_labels->points[*pit].x;
                float delta_y = p.y - cloud_filtered_labels->points[*pit].y;
                float delta_z = p.z - cloud_filtered_labels->points[*pit].z;

                float dist = delta_x * delta_x + delta_y*delta_y + delta_z*delta_z ;
                if (dist > dist_max){
                    dist_max = dist ;
                }
            }

            ClusterStruct cluster;
            cluster.label = (uint8_t) cl;
            cluster.x = p.x ;
            cluster.y = p.y ;
            cluster.z = p.z ;
            cluster.radius = std::sqrt(dist_max);
            copy(cluster_label_probs.begin(), cluster_label_probs.end(), back_inserter(cluster.label_probs));

            clusters.push_back(cluster);
        } //cl clusters
    } // cl
}// end Cluster Function


void ClustersPointClouds::computeBoW(const pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_in, std::vector<float> &boW_descriptor ){

    for( int i = 0 ; i < cloud_in->points.size() ; i++ ){
        if( cloud_in->points[i].label >= num_labels) continue;

        boW_descriptor[ cloud_in->points[i].label ]++ ;
    }

}// end bow Function


void ClustersPointClouds::computeBoWP( const std::vector<ClusterStruct> clusters, std::vector<float> &boWP_descriptor){
    int num_cluster = clusters.size() ;

    ROS_INFO_STREAM("Making BoWP with " << num_cluster << " clusters");

    vector<float>  array1D_init(num_labels ,0);
    vector<vector<float> >  cluster_pairs( num_labels , array1D_init );

    // Computing BOWP histogram by using eucleadean distance
    for( int i = 0 ; i < num_cluster ; ++i ){
        for( int j = i ; j < num_cluster ; ++j ){
            if (i != j ){
                float delta_x = clusters[i].x - clusters[j].x ;
                float delta_y = clusters[i].y - clusters[j].y ;
                float delta_z = clusters[i].z - clusters[j].z ;
                float dist = std::sqrt( delta_x * delta_x + delta_y*delta_y + delta_z*delta_z );

                if( clusters[i].radius > dist )
                    cluster_pairs[ clusters[i].label ][ clusters[j].label] += 1; // Incrementing previous value at that index of that vector by 1

                if( clusters[j].radius > dist )
                    cluster_pairs[ clusters[j].label ][ clusters[i].label] += 1; // Incrementing previous value at that index of that vector by 1

            } //end if main
        } // end i loop
    } // end j loop


    // Converting cluster_pairs into array1D
    for (int row = 0; row < num_labels; row++)
        for (int col = 0; col < num_labels ; col++)
            boWP_descriptor[ row*num_labels+col ] = cluster_pairs[row][col] ;
}// end boWP Function    


// Main clusters_node
int main(int argc, char **argv)
{
    ros::init( argc, argv, "clusters_node" );
    ClustersPointClouds cluster_point_cloud;
    ros::spin();
    return 0 ;
}
