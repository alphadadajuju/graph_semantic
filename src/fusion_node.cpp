// ROS
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

// ROS messages
#include <std_msgs/Int16.h>
#include "std_msgs/String.h"
#include "visualization_msgs/Marker.h"

#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32MultiArray.h"

#include "graph_semantic/FusedPointcloud.h"
#include "graph_semantic/RGB2LabelProb.h"

// Headers C_plus_plus
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <csignal>
#include <sstream>
#include <cmath>
#include <limits>
#include <float.h>
#include "boost/filesystem.hpp"

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/filters/frustum_culling.h>
#include <pcl/visualization/common/common.h>

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
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Geometry>


using namespace std;
//typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

void read_poses(const std::string& name, vector<Eigen::Matrix4f> &v){

    ifstream inFile;
    inFile.open(name);
    if (!inFile)
        ROS_ERROR_STREAM("Unable to open file datafile.txt");


    float val;
    while(inFile >> val){
        Eigen::Matrix4f pose;

        pose(0,0) = val;
        for (int i = 0; i < 3; ++i) { // row
            for (int j = 0; j < 4; ++j) { //col
                if (i == 0 && j == 0) continue;

                inFile >> val;
                pose(i,j) = val;
            }
        }
        // Fill in last row
        pose(3,0) = 0; pose(3,1) = 0; pose(3,2) = 0; pose(3,3) = 1;

        v.push_back(pose);
    }

}

void read_directory(const std::string& name, vector<string> &v) {

    try{
        boost::filesystem::path p(name);
        boost::filesystem::directory_iterator start(p);
        boost::filesystem::directory_iterator end;

        //GET files in directory
        vector<boost::filesystem::path> paths;
        std::copy(start, end, back_inserter(paths));

        //SORT them according to criteria
        struct sort_functor {
            //Return true if b bigger than a, false if equal or smaller
            bool operator()(const boost::filesystem::path &a, const boost::filesystem::path &b) {
                if(a.string().size() == b.string().size())
                    return a.compare(b) < 0;
                else
                    return a.string().size() < b.string().size();
            }
        };
        std::sort(paths.begin(), paths.end(), sort_functor());

        //OUTPUT vector of ordered strings
        for (vector<boost::filesystem::path>::const_iterator it(paths.begin()); it != paths.end(); ++it)
            v.push_back(it->string());

    }catch (const boost::filesystem::filesystem_error& ex)
    {
        cout << ex.what() << '\n';
    }

}

class SemanticFusion{

private:
    int num_labels;
    cv::Mat label2bgr;

    ros::NodeHandle nh;         // node handler
    ros::Publisher marker_pub ;   // Clusters node Message Publisher
    ros::Publisher pc_display_pub ;   // Clusters node Message Publisher
    ros::Publisher fused_pc_pub ;   // Clusters node Message Publisher
    ros::ServiceClient segnet_client;

    void matrix2multiarray(const vector<vector<float>> &label_probs, std_msgs::Float32MultiArray &msg_out);

    void displayCameraMarker(Eigen::Matrix4f cam_pose);
    void createRGBLabelsPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc, const vector<vector<float>> &label_prob,
                                   pcl::PointCloud<pcl::PointXYZL> &labelled_pc, pcl::PointCloud<pcl::PointXYZRGB> &label_rgb_pc);

    void displayPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc);

public:

    SemanticFusion();
    bool debug_mode;

    bool loadPlaceData(const string &base_path, pcl::PointCloud<pcl::PointXYZRGB> &pc, vector<Eigen::Matrix4f> &poses, vector<string> &images);
    void createFusedSemanticMap(const string &base_path, const pcl::PointCloud<pcl::PointXYZRGB> &pc, const vector<Eigen::Matrix4f> &poses, const vector<string> &images);

    void loadAndPublishLabelledPointcloud(string path);
    void storePointsProbDist(const string &base_path, const vector<vector<float> > &label_prob);

    void loadPointsProbDist(const string &label_probs_path, vector<vector<float>> &label_prob);
};


SemanticFusion::SemanticFusion(){
    debug_mode = true;

    num_labels = -1;
    ros::param::get("graph_semantic/num_labels", num_labels) ;

    // Read color to labels image
    string caffe_model_path;
    ros::param::get("graph_semantic/caffe_model_path", caffe_model_path);

    string label2bgr_filepath = caffe_model_path + "/sun_redux.png";
    label2bgr = cv::imread(label2bgr_filepath, CV_LOAD_IMAGE_COLOR);

    pc_display_pub = nh.advertise<sensor_msgs::PointCloud2>("graph_semantic/fusion_pointcloud", 10); // Publisher
    marker_pub = nh.advertise<visualization_msgs::Marker>("graph_semantic/camera_pose", 10); // Publisher
    fused_pc_pub = nh.advertise<graph_semantic::FusedPointcloud>("graph_semantic/semantic_fused_pc", 10); // Publisher

    ros::service::waitForService("rgb_to_label_prob", 12);
    segnet_client = nh.serviceClient<graph_semantic::RGB2LabelProb>("rgb_to_label_prob");
    ros::Duration(1).sleep();
}

bool
SemanticFusion::loadPlaceData(const string &base_path, pcl::PointCloud<pcl::PointXYZRGB> &pc, vector<Eigen::Matrix4f> &poses,
                              vector<string> &images) {

    ROS_INFO_STREAM("fusion_node: " << " processing directory " << base_path);

    boost::filesystem::path p(base_path);
    if(!boost::filesystem::exists(base_path))
        ROS_ERROR_STREAM("fusion_node: Invalid base-path for loading place data");


    string pc_path = base_path + "cloud.ply";
    pcl::io::loadPLYFile(pc_path, pc);

    string poses_path = base_path + "poses.txt";
    read_poses(poses_path, poses);

    string rgb_path = base_path + "rgb";
    read_directory(rgb_path, images);

    ROS_INFO_STREAM("fusion_node: " << " loaded cloud with " << pc.points.size() << " points.");
    ROS_INFO_STREAM("fusion_node: " << " loaded " << poses.size() << " poses.");
    ROS_INFO_STREAM("fusion_node: " << " loaded " << images.size() << " image's path.");

    return true;
}

void SemanticFusion::createFusedSemanticMap(const string &base_path, const pcl::PointCloud<pcl::PointXYZRGB> &pc,
                                            const vector<Eigen::Matrix4f> &poses, const vector<string> &images) {

    // Initialize label probability structure
    vector<float> label_prob_init(num_labels, 1.0f/(1.0f*num_labels));
    vector<vector<float> > label_prob(pc.points.size(), label_prob_init);

    // Initialize other objects
    pcl::FrustumCulling<pcl::PointXYZRGB> fc;
    fc.setInputCloud ( pc.makeShared() );
    // Set frustrum according to Kinect specifications
    fc.setVerticalFOV (45);
    fc.setHorizontalFOV (58);
    fc.setNearPlaneDistance (0.5);
    fc.setFarPlaneDistance (6); //Should be 4m. but we bump it up a little

    // Initialization of P
    Eigen::Matrix4f P;
    //P << 525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    P << 570.34222412109375, 0., 319.5, 0., 0.0, 570.34222412109375, 239.5, 0., 0., 0.0, 1., 0.0, 0.0, 0.0, 0.0, 1.0;

    // For each pose & image
    for (int i = 0; i < poses.size(); ++i) {
        ROS_INFO_STREAM_THROTTLE(5, "Processing node " << i << "/" << poses.size());

        if(!ros::ok()) return;

        // FrustrumCulling from poses-> indices of visible points
        Eigen::Matrix3f rot_xtion;
        rot_xtion = Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf( 0.0f*M_PI, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitX());

        Eigen::Matrix4f rot4 = Eigen::Matrix4f::Identity(4,4);
        rot4.topLeftCorner(3,3) = rot_xtion;

        Eigen::Matrix4f xtion_pose =  poses[i] * rot4.inverse();
        //xtion_pose.topLeftCorner(3,3) = rot_xtion * xtion_pose.topLeftCorner(3,3);


        fc.setCameraPose(xtion_pose);
        std::vector<int> inside_indices; // Indices of points in pc_gray inside camera frustrum at pose[i]
        fc.filter(inside_indices);

        // SRV: request image labelling through segnet
        cv::Mat cv_img = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);
        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg();

        graph_semantic::RGB2LabelProb srv_msg;
        srv_msg.request.rgb_image = *image_msg.get();
        segnet_client.call(srv_msg);

        //Display RGB image to match server's time of publication
        sensor_msgs::ImagePtr rgb_img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_img).toImageMsg();

        std_msgs::Float32MultiArray frame_label_probs(srv_msg.response.image_class_probability);
        int dim_stride_1 = frame_label_probs.layout.dim[1].stride;
        int dim_stride_2 = frame_label_probs.layout.dim[2].stride;


        //Create matrix to go from RVIZ depth camera frame to CV rgb camera frame
        Eigen::Matrix4f rviz2cv = Eigen::Matrix4f::Identity(4,4);
        Eigen::Matrix3f r;
        r = Eigen::AngleAxisf(0.0f*M_PI, Eigen::Vector3f::UnitZ())
            * Eigen::AngleAxisf( -0.5f*M_PI, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf( 0.5f*M_PI, Eigen::Vector3f::UnitX());
        rviz2cv.topLeftCorner(3,3) = r;

        Eigen::Matrix4f depth2optical = Eigen::Matrix4f::Identity(4,4);
        depth2optical(0,3) = -0.045f; //-0.025

        // Get camera matrix from poses_i and K
        Eigen::Matrix4f pose_inv = xtion_pose.inverse();
        Eigen::Matrix4f world2pix = P * depth2optical * rviz2cv * pose_inv;

        //Create "Z buffer" emulator
        int pixel_point_proj[480][640];
        float pixel_point_dist[480][640];
        fill_n(&pixel_point_proj[0][0], sizeof(pixel_point_proj) / sizeof(**pixel_point_proj), -1);
        fill_n(&pixel_point_dist[0][0], sizeof(pixel_point_dist) / sizeof(**pixel_point_dist), FLT_MAX);

        for(std::vector<int>::iterator it = inside_indices.begin(); it != inside_indices.end(); it++){
            // Backproject points using camera matrix (discard out of range)
            Eigen::Vector4f point_w(pc.points[*it].x, pc.points[*it].y, pc.points[*it].z, 1.0);
            Eigen::Vector4f point_px = world2pix * point_w;
            if(point_px(2) == 0){ point_px(2) +=1; } // Adapt homogenous for 4x4 efficient multiplication

            int pix_x = (int) std::round(point_px(0)/point_px(2));
            int pix_y = (int) std::round(point_px(1)/point_px(2));

            if(pix_x < 0 || pix_x >= 640 || pix_y < 0 || pix_y >= 480)
                continue;

            //Compute distance from camera position to point position
            Eigen::Vector3f cam_point_vec(xtion_pose(0,3)-pc.points[*it].x,
                                          xtion_pose(1,3)-pc.points[*it].y,
                                          xtion_pose(2,3)-pc.points[*it].z);
            float cam_point_dist = cam_point_vec.norm();

            //Check against tables and keep closest point's index
            if(cam_point_dist < pixel_point_dist[pix_y][pix_x]){
                pixel_point_proj[pix_y][pix_x] = *it;
                pixel_point_dist[pix_y][pix_x] = cam_point_dist;
            }
        }


        //Get associated distributions, multiply  distributions together and renormalize
        for (int y = 0; y < 480; ++y) {
            for (int x = 0; x < 640; ++x) {
                if(pixel_point_proj[y][x] == -1){
                    continue;
                }

                //Get idx of nearest projected point from pointcloud
                int pc_idx = pixel_point_proj[y][x];

                // Multiply each element of the distribution and Get the sum of the final distribution
                float prob_dist_sum = 0.0;
                for (int cl = 0; cl < num_labels; ++cl) {
                    label_prob[pc_idx][cl] *= frame_label_probs.data[dim_stride_1*y + dim_stride_2*x + cl];
                    prob_dist_sum += label_prob[pc_idx][cl];
                }

                // Divide each element by the sum so that they add up to 1
                for (int cl = 0; cl < num_labels; ++cl) {
                    label_prob[pc_idx][cl] /= prob_dist_sum;
                }
            }
        }

        if(debug_mode){
            pcl::PointCloud<pcl::PointXYZL> labelled_pc;
            pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
            createRGBLabelsPointcloud(pc, label_prob, labelled_pc, label_rgb_pc);

            displayCameraMarker(xtion_pose);
            displayPointcloud(label_rgb_pc);
        }
    }// For each pose

    pcl::PointCloud<pcl::PointXYZL> labelled_pc;
    pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
    createRGBLabelsPointcloud(pc, label_prob, labelled_pc, label_rgb_pc);

    // Display final pointcloud
    displayPointcloud(label_rgb_pc);

    //Store pointcloud and probabilities
    string labelled_cloud_path = base_path + "labelled_cloud.ply";
    string labelled_cloud_rgb_path = base_path + "labelled_cloud_rgb.ply";
    string label_probs_path = base_path + "label_probabilities.txt";

    pcl::io::savePLYFile(labelled_cloud_path, labelled_pc, true);
    pcl::io::savePLYFile(labelled_cloud_rgb_path, label_rgb_pc, true);
    storePointsProbDist(label_probs_path, label_prob);

    //Create pointcloud message
    sensor_msgs::PointCloud2 labelled_pc_msg;
    pcl::toROSMsg(labelled_pc, labelled_pc_msg);
    labelled_pc_msg.header.frame_id = "map";
    // Create probabilities message
    std_msgs::Float32MultiArray label_prob_msg;
    matrix2multiarray(label_prob, label_prob_msg);
    // Publish both
    graph_semantic::FusedPointcloud fused_pc_msg;
    fused_pc_msg.labelled_pc = labelled_pc_msg;
    fused_pc_msg.label_probs = label_prob_msg;
    fused_pc_pub.publish(fused_pc_msg);
}

void SemanticFusion::matrix2multiarray(const vector<vector<float> > &label_probs,
                                       std_msgs::Float32MultiArray &msg_out){

    // Initialize layout
    msg_out.layout.data_offset = 0;
    msg_out.layout.dim.push_back(std_msgs::MultiArrayDimension());
    msg_out.layout.dim.push_back(std_msgs::MultiArrayDimension());

    // Fill layout dimensions
    msg_out.layout.dim[0].label = "height";
    msg_out.layout.dim[0].size = (uint) label_probs.size();
    msg_out.layout.dim[0].stride = (uint) (label_probs.size() * label_probs[0].size());

    msg_out.layout.dim[1].label = "width";
    msg_out.layout.dim[1].size = (uint) label_probs[0].size();
    msg_out.layout.dim[1].stride = (uint) label_probs[0].size();

    msg_out.data.clear();
    msg_out.data.resize(msg_out.layout.dim[0].stride, -500.0f);
    for (int i = 0; i < label_probs.size(); ++i) {
        for (int j = 0; j < label_probs[0].size(); ++j) {
            msg_out.data[j + msg_out.layout.dim[1].stride*i] = label_probs[i][j];
        }
    }
}

void SemanticFusion::storePointsProbDist(const string &label_probs_path, const vector<vector<float> > &label_prob){
    ofstream myfile;
    myfile.open(label_probs_path);

    for (int i = 0; i < label_prob.size(); ++i){
        for (int j = 0; j < label_prob[0].size(); ++j){
            myfile << label_prob[i][j] << " ";
        }
        myfile << endl;
    }

    myfile.close();
}

void SemanticFusion::loadPointsProbDist(const string &label_probs_path, vector<vector<float> > &label_prob){
    // LABEL PROB MUST BE ALREADY INITIALISED!
    ifstream myfile;
    myfile.open(label_probs_path);

    for (int i = 0; i < label_prob.size(); ++i) {
        for (int j = 0; j < label_prob[0].size(); ++j) {
            myfile >> label_prob[i][j];
        }
    }

    myfile.close();
}

void SemanticFusion::displayCameraMarker(Eigen::Matrix4f cam_pose) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time();
    marker.ns = "graph_semantic";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = cam_pose(0, 3);
    marker.pose.position.y = cam_pose(1, 3);
    marker.pose.position.z = cam_pose(2, 3);

    Eigen::Matrix3f mat = cam_pose.topLeftCorner(3,3);
    Eigen::Quaternionf q(mat);

    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    marker.scale.x = 0.5;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.5;
    marker.color.g = 0.5;
    marker.color.b = 0.5;

    marker_pub.publish( marker );
}

void SemanticFusion::createRGBLabelsPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc,
                                               const std::vector<vector<float> > &label_prob,
                                               pcl::PointCloud<pcl::PointXYZL> &labelled_pc,
                                               pcl::PointCloud<pcl::PointXYZRGB> &label_rgb_pc)
{
    // Build XYZL pointcloud
    for(int i = 0; i < pc.points.size(); i++) {
        pcl::PointXYZL p;
        p.x = pc.points[i].x;
        p.y = pc.points[i].y;
        p.z = pc.points[i].z;

        int max_label = num_labels+1;

        float max_label_prob = 1.0f/(num_labels*1.0f);
        for (int cl = 0; cl < num_labels; ++cl) {
            if(label_prob[i][cl] > max_label_prob){
                max_label = cl;
                max_label_prob = label_prob[i][cl];
            }
        }

        p.label = (uint32_t) max_label;

        labelled_pc.push_back(p);
    }

    // DEBUG:: Build XYZRGB pointcloud
    for(int i = 0; i < labelled_pc.points.size(); i++) {
        pcl::PointXYZRGB p;
        p.x = labelled_pc.points[i].x;
        p.y = labelled_pc.points[i].y;
        p.z = labelled_pc.points[i].z;

        if(labelled_pc.points[i].label > num_labels){
            p.b = 0;
            p.g = 0;
            p.r = 0;
        }else{
            cv::Vec3b color = label2bgr.at<cv::Vec3b>(cv::Point(labelled_pc.points[i].label, 0));
            p.b = color.val[0];
            p.g = color.val[1];
            p.r = color.val[2];
        }
        label_rgb_pc.push_back(p);
    }
}

void SemanticFusion::displayPointcloud(const pcl::PointCloud<pcl::PointXYZRGB> &pc) {
    sensor_msgs::PointCloud2 pc_disp_msg;
    pcl::toROSMsg(*pc.makeShared(), pc_disp_msg);
    pc_disp_msg.header.frame_id = "map";

    pc_display_pub.publish(pc_disp_msg);
}

void SemanticFusion::loadAndPublishLabelledPointcloud(string  base_path) {
    string labelled_cloud_path = base_path + "labelled_cloud.ply";
    string labelled_cloud_rgb_path = base_path + "labelled_cloud_rgb.ply";
    string label_probs_path = base_path + "label_probabilities.txt";

    // Load labelled cloud
    pcl::PointCloud<pcl::PointXYZL> labelled_pc;
    pcl::io::loadPLYFile(labelled_cloud_path, labelled_pc);

    // Initialize label probability structure and load point semantic probabilities
    vector<float> label_prob_init(num_labels, -500.0f); //1.0f/(1.0f*num_labels));
    vector<vector<float> > label_prob(labelled_pc.points.size(), label_prob_init);
    loadPointsProbDist(label_probs_path, label_prob);

    //Load and display rgb semantic labels
    pcl::PointCloud<pcl::PointXYZRGB> label_rgb_pc;
    pcl::io::loadPLYFile(labelled_cloud_rgb_path, label_rgb_pc);
    displayPointcloud(label_rgb_pc);

    //Create pointcloud message
    sensor_msgs::PointCloud2 labelled_pc_msg;
    pcl::toROSMsg(labelled_pc, labelled_pc_msg);
    labelled_pc_msg.header.frame_id = "map";

    // Create probabilities message
    std_msgs::Float32MultiArray label_prob_msg;
    matrix2multiarray(label_prob, label_prob_msg);

    // Publish both
    graph_semantic::FusedPointcloud fused_pc_msg;
    fused_pc_msg.labelled_pc = labelled_pc_msg;
    fused_pc_msg.label_probs = label_prob_msg;
    fused_pc_pub.publish(fused_pc_msg);

    ROS_INFO_STREAM("Finished loading labelled pc and probabilities");
}

// heigh = 480, width = 640
// [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
//XTION
// [ 570.34222412109375, 0., 319.5, 0., 570.34222412109375, 239.5, 0., 0., 1. ]

/*
 # Given a 3D point [X Y Z]', the projection (x, y) of the point onto
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w

for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = pc1->begin(); it != pc1->end(); it++){
    //cout << it->x << ", " << it->y << ", " << it->z << endl;
}
*/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "semantic_fusion_node");

    //string base_path = "/home/albert/Desktop/room_test_processed/kitchen_with_floor/";
    string base_path;
    //ros::param::get("dir", base_path);

    vector<string> base_paths;
    base_paths.push_back(base_path);

    
    base_paths.push_back("/home/alpha/Kinect_bag/graph_semantic_data/room_test1/");
    /*
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/alpha_dim_any_angle");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/alpha_light_any_angle");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/kitchen_no_floor");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/kitchen_with_floor");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/omair_dim_any_angle");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/omair_light_any_angle");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/waji_dim_any_angle");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/waji_light_any_angle2");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/alpha_light_view_from_bed2");
    base_paths.push_back("/home/albert/Desktop/semantic_data/room_test_redux/alpha_light_view_from_groud");
    */

    bool load_existing = false;
    ros::param::get("load_labelled", load_existing);

    SemanticFusion sem_fusion;
    for(int i = 0; i < base_paths.size(); i++){
        sem_fusion = SemanticFusion();
        ros::param::get("debug", sem_fusion.debug_mode);

        base_path = base_paths.at(i);
        if(base_path.back() != '/') base_path += '/';

        ROS_INFO_STREAM("Labelling scan: " << base_path);

        if(load_existing){
            sem_fusion.loadAndPublishLabelledPointcloud(base_path);
        }else{
            pcl::PointCloud<pcl::PointXYZRGB> pc;
            vector<Eigen::Matrix4f> poses;
            vector<string> images;

            sem_fusion.loadPlaceData(base_path, pc, poses, images);
            ros::Duration(0.1).sleep();
            sem_fusion.createFusedSemanticMap(base_path, pc, poses, images);
        }
    }


    return 0 ;
}
