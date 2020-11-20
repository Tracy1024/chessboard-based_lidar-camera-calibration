#include <iostream>
#include <stdio.h>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/console/parse.h>
#include <pcl-1.8/pcl/filters/passthrough.h>
#include <pcl-1.8/pcl/ModelCoefficients.h>
#include <pcl-1.8/pcl/sample_consensus/model_types.h>
#include <pcl-1.8/pcl/features/board.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/boundary.h>
#include <pcl/search/kdtree.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/common/centroid.h>
#include <pcl/common/geometry.h>
#include <pcl/ml/kmeans.h>
#include <pcl/filters/statistical_outlier_removal.h>


using namespace std;

//============================================point cloud processing====================================================

//load point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr load_cloud(std::string path){

    //cloud initialization
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ (new pcl::PointCloud<pcl::PointXYZ>);
    //read pcd file as point cloud
    pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud_);
    return cloud_;
}

//voxel filter, filter in x-,y- and z-direction
pcl::PointCloud<pcl::PointXYZ>::Ptr pass_through_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, int x_min, int x_max, int y_min, int y_max, int z_min, int z_max){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_NoNAN (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_z (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x (new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*input_cloud, *cloud_NoNAN, indices);
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;
    // filter in Z direction(ground filter)
    pass.setInputCloud (cloud_NoNAN);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (z_min, z_max);
    pass.filter (*cloud_filtered_z);

    pass.setInputCloud (cloud_filtered_z);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (y_min, y_max);
    pass.filter (*cloud_filtered_y);

    pass.setInputCloud (cloud_filtered_y);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (x_min, x_max);
    pass.filter (*cloud_filtered_x);

    if(cloud_filtered_x->points.size()>0)
        cout << "pass through filter works well!" << endl;
    else
        cerr << "no more point after pass through filter, please adjust the parameters!" << endl;

    return cloud_filtered_x;
}

//calculate normal of nearst neibours, he neibours in the sphere with the radius
pcl::PointCloud<pcl::Normal>::Ptr calculate_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius){

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> estimator;
    estimator.setInputCloud(cloud);

    //use KdTree Search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    estimator.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    estimator.setRadiusSearch(radius);
    estimator.compute(*normals);

    if(normals->points.size()>0)
        cout << "normals has been founded!" << endl;
    else
        cerr << "can't find normals" << endl;

    return normals;
}

//segment the plane using RANSAC
pcl::PointCloud<pcl::PointXYZ>::Ptr seg_plane_RANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, int iteration_max, float threshold){

    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(iteration_max);
    seg.setDistanceThreshold(threshold);
    seg.setInputCloud(cloud);
    seg.setInputNormals(normal);

    // Obtain the plane inliers and coefficients
    seg.segment(*inliers_plane, *coefficients_plane);

    // Extract the planar inliers from the input cloud
    extract.setInputCloud(cloud);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);
    //filter to get plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    extract.filter (*cloud_plane);

    if(cloud_plane->points.size()>0)
        cout << "plane has been founded!" << endl;
    else
        cerr << "can't find plane" << endl;

    return cloud_plane;
}

//remove outliers using statistical outlier removal filter
pcl::PointCloud<pcl::PointXYZ>::Ptr statOutlierRemoval_filter (pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int num_neighbours, float dist_threshold){

    // Create the filtering object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (num_neighbours);
    sor.setStddevMulThresh (dist_threshold);
    sor.filter (*cloud_filtered);

    if(cloud_filtered->points.size()>0)
        cout << "statOutlierRemoval_filter works well!" << endl;
    else
        cerr << "no more point after statOutlierRemoval_filter, please adjust the parameters!" << endl;

    return cloud_filtered;

}

//calculate boundary
pcl::PointCloud<pcl::PointXYZ>::Ptr find_boundary(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius){

    //calculate normal
    pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
    normal = calculate_normal(cloud, radius);

    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> estimator;
    estimator.setInputCloud(cloud);
    estimator.setInputNormals(normal);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    estimator.setSearchMethod(tree);
    estimator.setKSearch(50);

    estimator.compute(boundary);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary(new pcl::PointCloud<pcl::PointXYZ>);
    for(size_t i = 0; i < cloud->points.size(); ++i) {
        if(boundary[i].boundary_point > 0) {
            cloud_boundary->push_back(cloud->points[i]);
        }
    }

    if(cloud_boundary->points.size()>0)
        cout << "boundary has been founded!" << endl;
    else
        cerr << "can't find boundary" << endl;

    return cloud_boundary;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr find_corner_candidates_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius){

    //kdtree find corner candidates
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    std::vector<int> rank_nn(80,1000);
    std::vector<int> nn_idx(80,0);


    for(size_t i = 0; i < cloud->points.size(); i++){

        std::vector<int>::iterator iter = max_element(rank_nn.begin(), rank_nn.end());
        int idx = std::distance(std::begin(rank_nn), iter);
        //std::cout << "idx: " << idx << std::endl;
        pcl::PointXYZ searchPoint = cloud->points[i];

        if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
        {
            int num_nb = pointIdxRadiusSearch.size();
            //std::cout << "number of nn " << num_nb << std::endl;
            int idx_cloud = nn_idx[idx];
            pcl::PointXYZ refPoint = cloud->points[idx_cloud];
            if(num_nb<*iter && (searchPoint.x!=refPoint.x || searchPoint.y!=refPoint.y || searchPoint.z!=refPoint.z)){
                rank_nn[idx] = num_nb;
                nn_idx[idx] = i;
            }
        }
        else
            continue;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ point;
    for(size_t i = 0;i<nn_idx.size();i++){

        cloud_out->push_back(cloud->points[nn_idx[i]]);

    }

    return cloud_out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr find_corner_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){

    //calculate distance of each point to centroid
    std::vector<float> dis_to_centroid(cloud->points.size(),0);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    pcl::PointXYZ centroid_;
    centroid_.x = centroid[0];
    centroid_.y = centroid[1];
    centroid_.z = centroid[2];

    for(size_t i = 0;i<cloud->points.size();i++){

        pcl::PointXYZ point_ = cloud->points[i];
        float dis_ = pcl::geometry::squaredDistance(point_, centroid_);
        dis_to_centroid[i] = dis_;

    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr corners (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ corner_candidate1;
    pcl::PointXYZ corner_candidate2;
    pcl::PointXYZ corner_candidate3;
    pcl::PointXYZ corner_candidate4;

    float dis_max1 = 0.1;
    float dis_max2 = 0.1;
    float dis_max3 = 0.1;
    float dis_max4 = 0.1;

    for(size_t i=0;i<cloud->points.size();i++){

        pcl::PointXYZ point_ = cloud->points[i];

        if((point_.x>centroid_.x)&&(point_.z>centroid_.z)){
            if(dis_to_centroid[i]>dis_max1) {
                corner_candidate1 = point_;
                dis_max1 = dis_to_centroid[i];
            }
        }

        if((point_.x<centroid_.x)&&(point_.z>centroid_.z)){
            if(dis_to_centroid[i]>dis_max2) {
                corner_candidate2 = point_;
                dis_max2 = dis_to_centroid[i];
            }
        }

        if((point_.x>centroid_.x)&&(point_.z<centroid_.z)){
            if(dis_to_centroid[i]>dis_max3) {
                corner_candidate3 = point_;
                dis_max3 = dis_to_centroid[i];
            }
        }

        if((point_.x<centroid_.x)&&(point_.z<centroid_.z)){
            if(dis_to_centroid[i]>dis_max4) {
                corner_candidate4 = point_;
                dis_max4 = dis_to_centroid[i];
            }
        }
    }
    corners->push_back(corner_candidate1);
    corners->push_back(corner_candidate2);
    corners->push_back(corner_candidate3);
    corners->push_back(corner_candidate4);

    return corners;

}

std::vector<cv::Point3f> generate_corners_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr corners, int corner_proRow, int corner_proCol){

    std::vector<cv::Point3f> corners3D;

    float dis_verti_x = (corners->points[0].x - corners->points[2].x)/(corner_proCol+1);
    float dis_verti_y = (corners->points[0].y - corners->points[2].y)/(corner_proCol+1);
    float dis_verti_z = (corners->points[0].z - corners->points[2].z)/(corner_proCol+1);

    float dis_hori_x = (corners->points[0].x - corners->points[1].x)/(corner_proRow+1);
    float dis_hori_y = (corners->points[0].y - corners->points[1].y)/(corner_proRow+1);
    float dis_hori_z = (corners->points[0].z - corners->points[1].z)/(corner_proRow+1);

    for(int i=1;i<corner_proCol+1;i++){

        float x_start =  corners->points[0].x - i*dis_verti_x;
        float y_start =  corners->points[0].y - i*dis_verti_y;
        float z_start =  corners->points[0].z - i*dis_verti_z;

        for(int j=1;j<corner_proRow+1;j++){
            float x_current = x_start - j*dis_hori_x;
            float y_current = y_start - j*dis_hori_y;
            float z_current = z_start - j*dis_hori_z;
            corners3D.push_back(cv::Point3f(x_current, y_current, z_current));

        }
    }

    return corners3D;

}


void visualisation_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud3){

    //show three point clouds with B,G,R
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler1(cloud1, 0, 0, 255);
    viewer->addPointCloud(cloud1, color_handler1,"cloud1");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler2(cloud2, 0, 255, 0);
    viewer->addPointCloud(cloud2, color_handler2,"cloud2");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler3(cloud3, 255, 0, 0);
    viewer->addPointCloud(cloud3, color_handler3,"cloud3");
    viewer->setCameraPosition(0, 0, -2, 0, -1, 0, 0);
    viewer->spin();

}


std::vector<cv::Point3f> corners3D_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_passthrough_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normal (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_boundary (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr corner_candidates (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr corner_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<cv::Point3f> corners_3D;

    cloud_passthrough_filtered = pass_through_filter(cloud, -4, 8, -13, 2, -2, 2);
    normal = calculate_normal(cloud_passthrough_filtered, 1.5);
    plane = seg_plane_RANSAC(cloud_passthrough_filtered, normal, 500, 0.05);
    plane_filtered = statOutlierRemoval_filter(plane, 30, 0.1);
    //plane_boundary = find_boundary(plane_filtered, 1.5);
    corner_candidates = find_corner_candidates_cloud(plane_filtered, 0.5);
    corner_cloud = find_corner_cloud(corner_candidates);
    corners_3D = generate_corners_cloud(corner_cloud, 6, 8);
    visualisation_cloud(cloud_passthrough_filtered, plane, corner_cloud);

    return corners_3D;

}



//============================================image processing==========================================================

//find corners in image

std::vector<cv::Point2f> find_corner_img( cv::Mat img, int corner_proRow, int corner_proCol) {

    cv::Mat Extractcorner;
    std::vector<cv::Point2f> corners;

    Extractcorner = img.clone();

    cv::Size board_size = cv::Size(corner_proRow, corner_proCol);

    int iter_adaptiv;

    int width, height;
    width = img.size().width;
    height = img.size().height;

    for (int i = 0; i < 3; i++) {

        bool patternfound = cv::findChessboardCorners(img, board_size, corners,
                                                      cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_ADAPTIVE_THRESH +
                                                      cv::CALIB_CB_NORMALIZE_IMAGE);

        if (!patternfound & i!=2) {
            cv::Rect area(ceil(width / (4 - i)), ceil(height / (4 - i)), ceil(width / (4 - i) * (2 - i)),
                          ceil(height / (4 - i) * (2 - i)));
            img = img(area);
            cv::imshow("test", img);
            cv::waitKey(0);

        }
        else if(patternfound){

            cv::Mat imageGray;
            cv::cvtColor(img, imageGray, CV_RGB2GRAY);
            cv::cornerSubPix(imageGray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            iter_adaptiv = max(0, i-1);
            break;
        }
        else{
            std::cout << "can not find chessboard corners!" << std::endl;
            exit(1);
        }
    }

    for (std::size_t i = 0; i < corners.size(); i++) {

        cout << corners[i].x << "   " << corners[i].y << endl;
    }

    return corners;

}

void visualization_corner2D( cv::Mat img, std::vector<cv::Point2f> corners){

    for (int i = 0; i < corners.size(); i++)
    {
        cv::circle(img, corners[i], 5, cv::Scalar(255, 0, 255), 2);
    }
    cv::imshow("Extractcorner", img);
    cv::waitKey(0);

}

void projection_box(pcl::PointCloud<pcl::PointXYZ>::Ptr corners3D, cv::Mat img, cv::Mat R, cv::Mat t, cv::Mat para_cameraIn, cv::Mat dist_Coeff){

    std::vector<cv::Point3f> pts_3d;
    for (size_t i = 0; i < corners3D->size(); ++i) {

        pcl::PointXYZ point_3d = corners3D->points[i];
        pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));

    }

    // project 3d-points into image view
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, R, t, para_cameraIn, dist_Coeff, pts_2d);


    //creat projected box
    cv::Rect box(pts_2d[0].x, pts_2d[0].y, abs(pts_2d[1].x-pts_2d[0].x), abs(pts_2d[2].y-pts_2d[0].y));
    cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2, cv::LINE_8, 0);

    cv::imshow("projeted box", img);
    cv::waitKey(0);
}

void project(pcl::PointCloud<pcl::PointXYZ>::Ptr corners3D, cv::Mat img, cv::Mat R, cv::Mat t, cv::Mat para_cameraIn, cv::Mat dist_Coeff){

    int deep_max = 0;
    std::vector<cv::Point3f> pts_3d;

    for (size_t i = 0; i < corners3D->size(); ++i) {

        pcl::PointXYZ point_3d = corners3D->points[i];
        pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));

    }

    for(size_t i = 0; i < pts_3d.size(); ++i){

        if(pts_3d[i].y > deep_max)
            deep_max = pts_3d[i].y;
    }

    // project 3d-points into image view
    std::vector<cv::Point2f> pts_2d;
    cv::projectPoints(pts_3d, R, t, para_cameraIn, dist_Coeff, pts_2d);

    int image_rows = img.rows;
    int image_cols = img.cols;
    cv::Mat img_projection = img.clone();

    for (size_t i = 0; i < pts_2d.size(); ++i){

        cv::Point2f point_2d = pts_2d[i];

        if (point_2d.x > 0 && point_2d.x < image_cols && point_2d.y > 0 && point_2d.y < image_rows)
        {
            // 0，1，2：B，G，R
            if(pts_3d[i].y < deep_max/3) {
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] = 0;
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] = 0;
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[2]= floor(pts_3d[i].y/(deep_max/3)*255);

            }
            else if(pts_3d[i].y < deep_max/3*2){
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] = floor(pts_3d[i].y/(deep_max/3*2)*255);
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] = 0;
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[2] = 0;
            }
            else{
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] = 0;
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] = 0;
                img_projection.at<cv::Vec3b>(point_2d.y, point_2d.x)[2] = floor(pts_3d[i].y/deep_max*255);
            }
        }
        else{
            continue;
        }
    }
}



int main() {

    //find corners in point cloud
    std::string path_pointcloud = "/home/yang/Documents/learn/multimodal_data_studio/20190909_blue_calib_experiment_sync_0.1_sample_25_data/lidar0/000003.pcd";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input (new pcl::PointCloud<pcl::PointXYZ>);
    cloud_input = load_cloud(path_pointcloud);

    std::vector<cv::Point3f> corners_3D;
    corners_3D = corners3D_cloud(cloud_input);



    //find corners in image
    cv::Mat image;
    std::string path_img = "/home/yang/Documents/learn/multimodal_data_studio/20190909_blue_calib_experiment_sync_0.1_sample_25_data/cam_front/000003.png";
    image = cv::imread(path_img);

    std::vector<cv::Point2f> corners_2D;
    corners_2D = find_corner_img(image, 6, 8);

    visualization_corner2D(image, corners_2D);



    //set camera in-parameters
    cv::Mat parameter_cameraIn = (cv::Mat_<float>(3, 3) << 1463.449562, 0.000000, 641.243665,
            0.000000, 1430.044716, 361.280449,
            0.000000, 0.000000, 1.000000);
    cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) <<  -0.472199, 0.042962, 0.000000, 0.000000, 0.0);

    //calculate ex-parameters
    cv::Mat rvecs, tvecs, inliers;
    cv::Mat rMat;



    //calculate ex-parameter
    cv::solvePnPRansac(corners_3D, corners_2D, parameter_cameraIn, distCoeffs, rvecs, tvecs, false, 100, 1.0, 0.99, inliers);
    cv::Rodrigues(rvecs, rMat);

    

//    std::cerr << "====================Result====================" << std::endl;
//    std::cout << "R_mat: " << rMat << std::endl;
//    std::cout << "t: " << tvecs << std::endl;

    return 0;

}





