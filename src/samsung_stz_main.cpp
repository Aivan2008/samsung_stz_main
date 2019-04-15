// SYSTEM  
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <mutex>
#include <list>
#include <map>
/////////////////////////////////////////////////////////////////////////
// OPENCV
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
/////////////////////////////////////////////////////////////////////////
// ROS  
#include <ros/ros.h>
#include "ros/package.h"
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/Twist.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <move_base_msgs/MoveBaseActionGoal.h>
#include "object_detection_msgs/DetectorResult.h"
#include <image_transport/image_transport.h>
/////////////////////////////////////////////////////////////////////////
// _____                               _                
//|  __ \                             | |               
//| |__) |_ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___ 
//|  ___/ _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|
//| |  | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \
//|_|   \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/
/////////////////////////////////////////////////////////////////////////
//Начальная фильтрация
//Процент высоты картинки выше которого мы объекты не рассматриваем как кубы
float min_y_position = 0.6f;
//Минимальная уверенность детектора ниже которой объекты не рассматриваются
float minimal_conf = 0.7f;
/////////////////////////////////////////////////////////////////////////
//Максимальная угловая скорость платформы, ограничение в целях безопасности
float max_angular_speed = 0.5f;
float min_angular_speed = 0.05f;
float angular_speed_multiplier = 0.5f;
//Максимальная линейная скорость платформы, ограничение в целях безопасности
float max_linear_speed = 0.8f;
float min_linear_speed = 0.05f;
float linear_speed_multiplier = 0.5f;
/////////////////////////////////////////////////////////////////////////
//TRACKING PARAMETERS
//То есть два кадра подряд надо найти объект рядом чтобы писать сообщения
int minimalTrajectoryLen = 2; 
//How to determine that curren cube is the same as last time?
//Минимально допустимое смещение (pixels) при котором сопровождение считается потерянным
int pixel_displacement_allowed = 20;
// Maximal allowed metric cube displacement between detections
float metric_displacement_allowed = 0.08;
bool useTracker = false;
const int followModeMoveBaseGoal = 1;
const int followModeControlByCoord = 2;
int followMode = followModeMoveBaseGoal;

/////////////////////////////////////////////////////////////////////////
//FOLLOW PARAMETERS
//Минмальное положение по вертикали ("желаемая" позиция куба по Y)
float vertical_desired_pos_rel = 0.98;
//Ho much time we should move to earlier detected object until say we lost it
int maximum_lost_frames = 500;
//Relative to image width, width of corridor where cube must be lost to be gathered
float gathering_area_relative_width = 0.2;

//Длина доезда в метрах
float follow_meter_length = 0.2;
//Допустимое смещение от целевого положения по прибытию в точку сбора
float gathering_delta_pos_allowed = 0.05;
//Допустимое смещение по углу от целевого объекта
float gathering_delta_angle_allowed = 0.05;
/////////////////////////////////////////////////////////////////////////
// ______                _   _                 
//|  ____|              | | (_)                
//| |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
//|  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
//| |  | |_| | | | | (__| |_| | (_) | | | \__ \
//|_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
/////////////////////////////////////////////////////////////////////////
//CALLBACKS
//Коллбэк на прием картинок
void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg);
//Коллбэк на прием рамок кубов
void detectorCallback(const object_detection_msgs::DetectorResultConstPtr& msg);
//Прием данных от одометрии
void OdomCallback(const nav_msgs::Odometry::ConstPtr& msg);
//Прием данных командного топика
void CommandCallback(const std_msgs::String::ConstPtr& msg);
//Прием данных от захвата
void grappleCallback(const std_msgs::Bool::ConstPtr& msg);
/////////////////////////////////////////////////////////////////////////
//FUNCTIONS
void initParams(ros::NodeHandle* nh_p);
void readParams(ros::NodeHandle* nh_p);
//Служебная функция для расчета положения куба в пространстве
void calcBoxPosition(double u, double v, double mod, double &X, double&Z, double &angle);
//
visualization_msgs::Marker GenerateMarker( int secs, int nsecs, int id, double x, double y, double scale, float r, float g, float b, float a);
//
move_base_msgs::MoveBaseActionGoal GenerateGoal(int sec, int nsec, double dest_x, double dest_y, double dest_angle);
/////////////////////////////////////////////////////////////////////////
// _____        _        
//|  __ \      | |       
//| |  | | __ _| |_ __ _ 
//| |  | |/ _` | __/ _` |
//| |__| | (_| | || (_| |
//|_____/ \__,_|\__\__,_|                     
/////////////////////////////////////////////////////////////////////////
//Текущее изображение для получения из коллбэка через мьютекс
cv::Mat currentImage;
std::mutex imageMutex;
//Время получения картинки (по времени ПК робота!)
int img_sec;
int img_nsec;
/////////////////////////////////////////////////////////////////////////
//Данные детектирования и мьютекс доступа к ним
std::mutex detectionDataMutex;
std::vector<std::vector<float> > receivedRects;
//Robot position in time of detection
float detectionRobotX = 0.0f;
float detectionRobotY = 0.0f;
float detectionRobotYaw = 0.0f;
//Время получения КАРТИНКИ на которых было осуществлено обнаружение
//Время ПО ВРЕМЕНИ РОБОТА!
int det_sec;
int det_nsec;
//Переменные для определения изменений в разнице по времени между
//временем кадра и детектора
//float delta_img_det=0.0f;
//float delta_img_det_prev=0.0f;
/////////////////////////////////////////////////////////////////////////
//ODOMETRY
std::mutex odometryDataMutex;
float odometryX = 0.0f;
float odometryY = 0.0f;
float odometryYaw = 0.0f;
double odometryTimeStamp=0.0;
/////////////////////////////////////////////////////////////////////////
//GRAPPLE
std::mutex grappleStateMutex;
bool grappleHoldCube = false;
/////////////////////////////////////////////////////////////////////////
//TRACKING AND FOLLOW
std::vector<double> previousDesiredObjectPosition;
std::vector<double> currentDesiredObjectPosition;
int desiredObjectTrajectoryLength=0;
/////////////////////////////////////////////////////////////////////////
//STATE MACHINE
//Search for cubes and send the message as confident cube found
const int STATE_SEARCH = 0;
//Here we follow the cube using its pixel or metric coords, if cube is lost or no detection we still move to coords and hope to find new one in almost same place
const int FOLLOW_CUBE = 1;
//Here we move forward until reach cube or pass desired length
const int GATHER_LOST_CUBE = 2;
std::mutex currentStateMutex;
int current_state = 0;
/////////////////////////////////////////////////////////////////////////
//Служебные РОСовские представители узла
ros::NodeHandle* nh;
ros::NodeHandle* nh_p;
/////////////////////////////////////////////////////////////////////////
// __  __       _       
//|  \/  |     (_)      
//| \  / | __ _ _ _ __  
//| |\/| |/ _` | | '_ \ 
//| |  | | (_| | | | | |
//|_|  |_|\__,_|_|_| |_|
/////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv )
{
  //Инициализация узла
  ros::init(argc, argv, "samsung_stz_main");
  ros::start();
  ROS_INFO("Ros started!");
  //Создать службу управления нодой
  nh = new ros::NodeHandle();
  nh_p = new ros::NodeHandle("~");
  // Сообщения для командного топика
  std_msgs::String msg_box_detected = std_msgs::String();
  std_msgs::String msg_box_taken = std_msgs::String();
  std_msgs::String msg_box_not_taken = std_msgs::String();
  msg_box_detected.data = std::string("GATHER_CUBE");
  msg_box_taken.data = std::string("GUBE_TAKEN");
  msg_box_not_taken.data = std::string("CUBE_NOT_TAKEN");
  //Создать паблишеров и сабскрайберов
  ros::Subscriber sub_image = nh->subscribe("/usb_cam_front/image_raw/compressed", 1, imageCallback);
  ros::Subscriber sub_detector_res = nh->subscribe("/samsung/BBoxes", 1, detectorCallback);
  ros::Subscriber sub_odom = nh->subscribe("/kursant_driver/odom", 1, OdomCallback);
  ros::Publisher twistPublisher = nh->advertise<geometry_msgs::Twist>("/kursant_driver/cmd_vel", 100);
  ros::Publisher commandPublisher = nh->advertise<std_msgs::String>("/kursant_driver/command", 100);
  ros::Publisher cubesPublisher = nh->advertise<visualization_msgs::MarkerArray>("/samsung/cube_positions", 20);
  ros::Publisher moveBaseGoalPublisher = nh->advertise<move_base_msgs::MoveBaseActionGoal>("/move_base/goal", 20);//
  ros::Publisher debugGoalPosePublisher = nh->advertise<geometry_msgs::PoseStamped>("/samsung/goal_pose", 20);
  image_transport::ImageTransport it(*nh);
  image_transport::Publisher debugImagePublisher = it.advertise("/samsung_stz_main/debug_image", 1);
  //Служебный параметр для отключения отправки движения/цели при отладке
  nh->setParam("/samsung_stz_main/move", 0);
  //Может быть полезно если решим что-то сохранять в файлы или загружать
  //std::string package_path = ros::package::getPath("tld_tracker_sams")+std::string("/lib/");
  //if(!package_path.empty())
  //  std::cout<<"Package path: "<<package_path.c_str()<<"\n";
  //Создать окно отображения информации об объектах и сопровождении
  //cv::namedWindow("view");
  //cv::startWindowThread();
  //Протинциализировать параметры из xml-launch файла
  initParams(nh_p);
  //Flag, if object we detected is confident (tracked successfully for more than minimalTrajectoryLen)
  bool detectedConfidentObject=false;
  //Flag, if there exists confident object that can be used for initialization
  bool new_object_found = false;

/////////////////////////////////////////////////////////////////////////////////////
//   _____           _      
//  / ____|         | |     
// | |    _   _  ___| | ___ 
// | |   | | | |/ __| |/ _ \
// | |___| |_| | (__| |  __/
//  \_____\__, |\___|_|\___|
//         __/ |            
//        |___/             
/////////////////////////////////////////////////////////////////////////////////////
  while(ros::ok())
  {
    //Считать параметры на случай если они изменились
    readParams(nh_p);

    //First of all - let's check grapple state
    grappleStateMutex.lock();
    bool grapple_hold_cube = grappleHoldCube;
    grappleStateMutex.unlock();

    //Получить блокировку, извлечь текущую картинку для отображения
    //Получить также таймстамп картинки для сравнения с таймстампами детектора
    cv::Mat debug_img, tracking_img;
    double image_tstamp = 0;
    imageMutex.lock();
    debug_img = currentImage.clone();
    int img_secs = img_sec;
    int img_nsecs=img_nsec;
    image_tstamp = (double)img_sec + (double)img_nsec/1000000000;
    imageMutex.unlock();
    ///////////////////////////////
    //Без картинки все остальное - бессмысленно
    if(!debug_img.data)
    {
        ros::spinOnce();
        continue;
    }
    //Отдельная копия - для отладочного отображения
    tracking_img = debug_img.clone();
    double calib_mod = 800.0/static_cast<double>(debug_img.cols);
    ///////////////////////////////
    //Получить блокировку, забрать данные о детекциях, очистить, чтобы старые
    //квадраты не ползали по экрану стоя на месте
    double detector_tstamp=0;
    //int detector_sec=0, detector_nsec=0;
    std::vector<std::vector<float> > detections;
    detectionDataMutex.lock();
    //detector_sec = det_sec;
    //detector_nsec = det_nsec;
    detector_tstamp = (double)det_sec + (double)det_nsec/1000000000;
    detections = receivedRects;
    receivedRects.clear();
    double det_x=0, det_y=0, det_yaw=0;
    det_x = detectionRobotX;
    det_y = detectionRobotY;
    det_yaw = detectionRobotYaw;
    detectionDataMutex.unlock();
    bool detection_received = detections.size()>0;
  
    double odom_x=0;
    double odom_y=0;
    double odom_yaw=0;
    double odom_stamp=0;
    odometryDataMutex.lock();
    odom_x = odometryX;
    odom_y = odometryY;
    odom_yaw = odometryYaw;
    odom_stamp = odometryTimeStamp;
    odometryDataMutex.unlock();
    

    //Расчет разницы по времени между детекцией и текущим кадром
    //delta_img_det = image_tstamp - detector_tstamp;
    ///////////////////////////////
    //!Отладочная инфа рисуется всегда
    //Исходя из заданных в параметрах отосительных координат рассчитать положение "линии горизонта" и
    //"целевой" линии куда мы ведем низ куба
    int minimal_y = int(min_y_position*debug_img.rows);
    //std::cout<<"Myp="<<min_y_position<<" my="<<minimal_y<<" rows="<<debug_img.rows<<"\n";
    int desired_y = int(vertical_desired_pos_rel*debug_img.rows);
    //int desired_x = debug_img.cols/2;
    int desired_area_left = (int)((float)(debug_img.cols)/2 - gathering_area_relative_width*(float)(debug_img.cols));
    int desired_area_right = (int)((float)(debug_img.cols)/2 + gathering_area_relative_width*(float)(debug_img.cols));
    //Отрисовать отладочные линии - "цель" по вертикали, горизонтали
    cv::line(debug_img, cv::Point2f(0, minimal_y), cv::Point2f(debug_img.cols-1, minimal_y), cv::Scalar(150,0,0), 1);
    cv::line(debug_img, cv::Point2f(0, desired_y), cv::Point2f(debug_img.cols-1, desired_y), cv::Scalar(0,150,0), 1);
    //cv::line(debug_img, cv::Point2f(desired_x, 0), cv::Point2f(desired_x, debug_img.rows-1), cv::Scalar(0,150,0), 1);
    cv::line(debug_img, cv::Point2f(desired_area_left, 0), cv::Point2f(desired_area_left, debug_img.rows-1), cv::Scalar(0,150,0), 1);
    cv::line(debug_img, cv::Point2f(desired_area_right, 0), cv::Point2f(desired_area_right, debug_img.rows-1), cv::Scalar(0,150,0), 1);

    visualization_msgs::MarkerArray cubes;
    std::vector<double> desired_rect;
    if(detection_received)
    {
        //Расчет пространственных координат куба для каждого прямоугольника
        //Опубликовать расчетные позы кубов
        std::vector<std::vector<double> > current_detections;
        for(int i=0; i<detections.size(); i++)
        {
            double X1,Z1,angle1, X2, Z2, angle2;
            float xmin = detections[i][0];
            float ymin = detections[i][1];
            float xmax = detections[i][2];
            float ymax = detections[i][3];
            float conf = detections[i][4];

  
            if(ymin>minimal_y && conf>minimal_conf)
            {
              std::vector<double> det;
              det.resize(7);
              calcBoxPosition(xmin, ymax, calib_mod, X1, Z1, angle1);
              calcBoxPosition(xmax, ymax, calib_mod, X2, Z2, angle2);

              //Дополнительно - усреднить между правым нижним и левым нижним
              double X = (X1+X2)/2;
              double Z = (Z1+Z2)/2;
              double A = (angle1+angle2)/2;

              cv::rectangle(debug_img, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(255,0,0), 2);
              std::stringstream ss;
              ss.setf(std::ios::fixed);
              ss.precision(2);
              ss<<"X="<<X<<" Z="<<Z;
              //Тут печатаем на картинке пространственные координаты
              cv::putText(debug_img, ss.str(), cvPoint(xmin,ymin-10),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255,0,0), 1, CV_AA);        

              //r[0] = X;
              //r[1] = Z;
              //r[2] = A;
              double map_x = det_x + (Z+0.20)*cos(det_yaw) + X*sin(det_yaw);
              double map_y = det_y + (Z+0.20)*sin(det_yaw) - X*cos(det_yaw);
              //r[3] = map_x;
              //r[4] = map_y;
              
              det[0] = xmin;
              det[1] = ymin;
              det[2] = xmax;
              det[3] = ymax;
              det[4] = map_x;
              det[5] = map_y;
              det[6] = A;
              current_detections.push_back(det);
            }
        }
        //Выбор того, к которому будем ехать
        currentDesiredObjectPosition.clear();
        float low_y=0.0f;
        //We had previous object, look if any detection is near
        bool prev_obj=false;
        float prev_u=0, prev_v=0, prev_x=0, prev_y=0;
        if(previousDesiredObjectPosition.size()>0)
        {
            prev_obj = true;
            prev_u = (previousDesiredObjectPosition[0]+previousDesiredObjectPosition[2])/2;
            prev_v = (previousDesiredObjectPosition[1]+previousDesiredObjectPosition[3])/2;
            prev_x = previousDesiredObjectPosition[4];
            prev_y = previousDesiredObjectPosition[5];
        }

        for(int i=0; i<current_detections.size(); i++)
        {
            float xmin = current_detections[i][0];
            float ymin = current_detections[i][1];
            float xmax = current_detections[i][2];
            float ymax = current_detections[i][3];

            float curr_u = (xmin+xmax)/2;
            float curr_v = (ymin+ymax)/2;
            float curr_x = current_detections[i][4];
            float curr_y = current_detections[i][5];
            float curr_a = current_detections[i][6];
  
            visualization_msgs::Marker cb = GenerateMarker( img_secs, img_nsecs, i, curr_x, curr_y, 0.1,204.0/255.0,168.0/255,45.0/255.0, 1.0);
            cubes.markers.push_back(cb);


            if(prev_obj && currentDesiredObjectPosition.empty())
            {
                double du = fabs(curr_u-prev_u);
                double dv = fabs(curr_v-prev_v);
                double dx = fabs(curr_x-prev_x);
                double dy = fabs(curr_y-prev_y);
                //std::cout<<"du = "<<du<<" dv = "<<dv<<" dx = "<<dx<<" dy = "<<dy<<"\n";
                if((fabs(curr_u-prev_u)<=pixel_displacement_allowed &&
                    fabs(curr_v-prev_v)<=pixel_displacement_allowed) ||
                    (fabs(curr_x-prev_x)<=metric_displacement_allowed &&
                    fabs(curr_y-prev_y)<=metric_displacement_allowed))
                {
                    std::cout<<"Tracked suceccfully!\n";
                    currentDesiredObjectPosition =  current_detections[i];
                    desiredObjectTrajectoryLength+=1;
                    continue;
                }
            }

            if(ymax>low_y)
            {
                desired_rect =  current_detections[i];
                low_y = ymax;
            }  
        }

        //Now we check if we
        if((currentDesiredObjectPosition.size()>0) && (desiredObjectTrajectoryLength>minimalTrajectoryLen))
        {
            //We gonna use this flag to send message that object foud
            detectedConfidentObject=true;
        }
        else
        {
            detectedConfidentObject=false;
        }
        //Set if we found at least one good confident object
        new_object_found = desired_rect.size()>0;
    }
   


    std::vector<double> currentTrackedRect;
    if(!detection_received)
    {
      if(useTracker)
      {
        //Here can be some tracking code, that allow us to track object while we are not receive detections
        int trc=0;
      }
      else
      {
        if(currentDesiredObjectPosition.size()>0)
          currentTrackedRect = currentDesiredObjectPosition;
      }
    }
    else
    {
      if(currentDesiredObjectPosition.size()>0)
        currentTrackedRect = currentDesiredObjectPosition;
    }

    currentStateMutex.lock();
    int state = current_state;
    currentStateMutex.unlock();
    std::stringstream state_line;
    static int lost_counter=0;

    static move_base_msgs::MoveBaseActionGoal gathering_goal = move_base_msgs::MoveBaseActionGoal();

    if(state == STATE_SEARCH)
    {
          state_line<<"State: search, ";
          //We found confident object, send message
          if(detection_received)
          {
              if(detectedConfidentObject)
              {
                  state_line<<"Pub command ";
                  commandPublisher.publish(msg_box_detected);
                  previousDesiredObjectPosition = currentDesiredObjectPosition;
                  cv::rectangle(debug_img, cv::Point2f(currentDesiredObjectPosition[0], currentDesiredObjectPosition[1]), cv::Point2f(currentDesiredObjectPosition[2],currentDesiredObjectPosition[3]), cv::Scalar(255,255,255), 2);
                  lost_counter=0;
                  currentStateMutex.lock();
                  current_state = FOLLOW_CUBE;
                  currentStateMutex.unlock();
                  visualization_msgs::Marker cb = GenerateMarker(img_secs,img_nsecs, detections.size()+1, previousDesiredObjectPosition[4], previousDesiredObjectPosition[5], 0.11, 1.0, 1.0, 1.0, 0.9);
                  cubes.markers.push_back(cb);
              }
              else
              {
                //We have lost object or had no-one, but now found a new one, switch to it then
                if(currentDesiredObjectPosition.size()>0)
                {
                    state_line<<"Update object ";
                    previousDesiredObjectPosition = currentDesiredObjectPosition;
                    cv::rectangle(debug_img, cv::Point2f(currentDesiredObjectPosition[0], currentDesiredObjectPosition[1]), 
                                  cv::Point2f(currentDesiredObjectPosition[2],currentDesiredObjectPosition[3]), cv::Scalar(255,255,255), 2);
                    visualization_msgs::Marker cb = GenerateMarker(img_secs,img_nsecs,detections.size()+1, previousDesiredObjectPosition[4], previousDesiredObjectPosition[5], 0.11, 0.5, 1.0, 1.0, 0.9);
                    cubes.markers.push_back(cb);
                }
                else
                {
                    if(new_object_found)
                    {
                      state_line<<"Reset object ";
                      previousDesiredObjectPosition = desired_rect;
                      desiredObjectTrajectoryLength = 1;
                      visualization_msgs::Marker cb = GenerateMarker(img_secs, img_nsecs, detections.size()+1,previousDesiredObjectPosition[4],previousDesiredObjectPosition[5],0.11, 1.0,1.0,1.0, 0.9);
                      cubes.markers.push_back(cb);
                    }
                    else //We lost an object and have no new one, as we do not move to it, drop it
                    {
                      state_line<<"Delete object\n";
                      previousDesiredObjectPosition.clear();
                    }
                }
              }
           }
           else
           {
              state_line<<"no new object ";
           }
    }
    else if(state==FOLLOW_CUBE)
    {

      state_line<<"State: follow, ";      //Мы доехали, сменить состояние
      if(grapple_hold_cube)
      {
        state_line<<"cube grappled ";
        currentStateMutex.lock();
        current_state = STATE_SEARCH;
        currentStateMutex.unlock(); 
        commandPublisher.publish(msg_box_taken);
        if(followMode==followModeMoveBaseGoal)
        {
          move_base_msgs::MoveBaseActionGoal goal_reset = GenerateGoal(img_secs, img_nsecs, odom_x, odom_y, odom_yaw);
          moveBaseGoalPublisher.publish(goal_reset);
          ros::spinOnce();
        }
        else
        {
          geometry_msgs::Twist twist;
          twistPublisher.publish(twist);  
          ros::spinOnce();
        }
      }
      //Here we designate an object to which we want t0
      //We found confident object, send message
      std::vector<double> destination_object;
      static cv::Scalar last_color = cv::Scalar(0, 255, 0);
      if(detection_received)
      {
        if(detectedConfidentObject)
        {
          state_line<<"Update confident ";
          commandPublisher.publish(msg_box_detected);
          previousDesiredObjectPosition = currentDesiredObjectPosition;
          destination_object = currentDesiredObjectPosition;
          cv::rectangle(debug_img, cv::Point2f(currentDesiredObjectPosition[0], currentDesiredObjectPosition[1]), 
                        cv::Point2f(currentDesiredObjectPosition[2],currentDesiredObjectPosition[3]), cv::Scalar(0,255,0), 2);
          last_color = cv::Scalar(0,255,0);
          visualization_msgs::Marker cb = GenerateMarker( img_secs, img_nsecs, detections.size()+1, previousDesiredObjectPosition[4], previousDesiredObjectPosition[5], 0.11, 0.0, 1.0, 0.0, 0.9);
          cubes.markers.push_back(cb);
          lost_counter=0;
        }
        else
        {
          //We have lost object or had no-one, but now found a new one, switch to it then
          if(currentDesiredObjectPosition.size()>0)
          {
            state_line<<"Update unconfident ";
            previousDesiredObjectPosition = currentDesiredObjectPosition;
            destination_object = previousDesiredObjectPosition;
            cv::rectangle(debug_img, cv::Point2f(currentDesiredObjectPosition[0], currentDesiredObjectPosition[1]), 
                        cv::Point2f(currentDesiredObjectPosition[2],currentDesiredObjectPosition[3]), cv::Scalar(0,255,0), 2);
            last_color = cv::Scalar(0,255,0);
            visualization_msgs::Marker cb = GenerateMarker( img_secs, img_nsecs, detections.size()+1, previousDesiredObjectPosition[4], previousDesiredObjectPosition[5], 0.11, 0.0, 0.5, 0.0, 0.9);
            cubes.markers.push_back(cb);
            lost_counter=0;
          }
          else
          {
            lost_counter+=1;
            state_line<<", lost_counter="<<lost_counter<<" ";
            if(lost_counter>=maximum_lost_frames)
            {
              //Тут есть варианты
              //Мы ехали-ехали и потеряли объект, не видим его на том же месте дольше какого-то времени, разумно большого
              //Потрачено, нет ни текущего ни предыдущего положения, куб потерян
              state_line<<"Pub LOST COUNTER ";
              commandPublisher.publish(msg_box_not_taken);
              ros::spinOnce();
              currentStateMutex.lock();
              current_state = STATE_SEARCH;
              currentStateMutex.unlock();
              if(followMode==followModeMoveBaseGoal)
              {
                move_base_msgs::MoveBaseActionGoal goal_reset = GenerateGoal(img_secs, img_nsecs, odom_x, odom_y, odom_yaw);
                moveBaseGoalPublisher.publish(goal_reset);
                ros::spinOnce();
              }
              else
              {
                geometry_msgs::Twist twist;
                twistPublisher.publish(twist);  
                ros::spinOnce();
              }
            }
            else
            {
              if(previousDesiredObjectPosition.size()>0)
              {
                state_line<<"Use last position ";
                destination_object = previousDesiredObjectPosition;
                cv::rectangle(debug_img, cv::Point2f(previousDesiredObjectPosition[0], previousDesiredObjectPosition[1]), 
                        cv::Point2f(previousDesiredObjectPosition[2],previousDesiredObjectPosition[3]), cv::Scalar(0,0,255), 2);
                last_color = cv::Scalar(0,0,255);
                visualization_msgs::Marker cb = GenerateMarker( img_secs, img_nsecs, detections.size()+1, previousDesiredObjectPosition[4], previousDesiredObjectPosition[5], 0.11, 1.0, 0.0, 0.0, 0.9);
                cubes.markers.push_back(cb);
                //Если предыдущее положение было в зоне захвата, то перейти в режим доезжания, иначе продолжить движение по алгоритму
              }
              else
              {
                //Потрачено, нет ни текущего ни предыдущего положения, куб потерян
                state_line<<"Pub LOST NO PREVIOUS DETECTION ";
                commandPublisher.publish(msg_box_not_taken);
                ros::spinOnce();
                currentStateMutex.lock();
                current_state = STATE_SEARCH;
                currentStateMutex.unlock(); 
                if(followMode==followModeMoveBaseGoal)
                {
                  move_base_msgs::MoveBaseActionGoal goal_reset = GenerateGoal(img_secs, img_nsecs, odom_x, odom_y, odom_yaw);
                  moveBaseGoalPublisher.publish(goal_reset);
                  ros::spinOnce();
                }
                else
                {
                  geometry_msgs::Twist twist;
                  twistPublisher.publish(twist);  
                  ros::spinOnce();
                }              
              }
            }
          }
        }
      }
      else
      {

        if(previousDesiredObjectPosition.size()>0)
        {
          //Это тоже ситуация, когда мы едем вслепую, просто либо обнаружения не пришли, это тут, либо пришли но там нет нашего объекта
          lost_counter+=1;
          state_line<<"no frame, use last position ";
          destination_object = previousDesiredObjectPosition;
          cv::rectangle(debug_img, cv::Point2f(previousDesiredObjectPosition[0], previousDesiredObjectPosition[1]), 
                  cv::Point2f(previousDesiredObjectPosition[2],previousDesiredObjectPosition[3]), last_color, 2);
          visualization_msgs::Marker cb = GenerateMarker( img_secs, img_nsecs, detections.size()+1, previousDesiredObjectPosition[4], previousDesiredObjectPosition[5], 0.11, 0.0, 1.0, 0.0, 0.9);
          cubes.markers.push_back(cb);

        }
      }
                              
      static move_base_msgs::MoveBaseActionGoal goal = move_base_msgs::MoveBaseActionGoal();
      if(destination_object.size()>0 )
      {
      
        //Если предыдущее положение было в зоне захвата, то перейти в режим доезжания, иначе продолжить движение по алгоритму
        double ymax = destination_object[3];
        double xmin = destination_object[0];
        double xmax = destination_object[2];
        if(xmin>xmax)
        {
          double temp = xmin;
          xmin=xmax;
          xmax = temp;
        }       

        //Переключение состояния по нижней границе
        if((xmin>desired_area_left) && (xmax<desired_area_right) && (ymax>=(desired_y)))
        {
          double next_x = odom_x + follow_meter_length*cos(odom_yaw);
          double next_y = odom_y + follow_meter_length*sin(odom_yaw);

          move_base_msgs::MoveBaseActionGoal goal2 = GenerateGoal(img_secs, img_nsecs, next_x, next_y, odom_yaw);
        }
    
        state_line<<", Destination exists ";
        //cv::rectangle(debug_img, cv::Point2f(destination_object[0], destination_object[1]), 
        //                cv::Point2f(destination_object[2],destination_object[3]), cv::Scalar(0,255,0), 2)
    
        //Рассчитываем дельту и угол
        static double dest_x = 0;
        static double dest_y = 0;
        static double delta_x = 0;
        static double delta_y = 0;
        static double dest_angle = 0;
        static double delta_angle = 0;
        dest_x = destination_object[4];
        dest_y = destination_object[5];// + follow_meter_length*sin(odom_yaw);
        delta_x = dest_x - odom_x;
        delta_y = dest_y - odom_y;
        dest_angle = atan2(delta_y, delta_x);
        dest_x+=follow_meter_length*cos(dest_angle);
        dest_y+=follow_meter_length*sin(dest_angle);
        delta_angle = dest_angle-odom_yaw;
        delta_angle+=(delta_angle>M_PI) ? -M_PI*2 : (delta_angle<-M_PI) ? 2*M_PI : 0; 
        if(detection_received) 
        { 
          goal = GenerateGoal(img_secs, img_nsecs, dest_x, dest_y, dest_angle);
        }
        geometry_msgs::PoseStamped gp;
        gp.header = goal.goal.target_pose.header;
        gp.pose = goal.goal.target_pose.pose;
        debugGoalPosePublisher.publish(gp);
        ros::spinOnce();

        if(fabs(delta_x)<gathering_delta_pos_allowed && fabs(delta_y)<gathering_delta_pos_allowed && fabs(delta_angle)<gathering_delta_angle_allowed)
        {
          //Считаем что прибыли в точку назначения, раз куб не захватили, то его и нет
          state_line<<" finish move, not found, end ";
          commandPublisher.publish(msg_box_not_taken);
          ros::spinOnce();
          currentStateMutex.lock();
          current_state = STATE_SEARCH;//GATHER_LOST_CUBE
          currentStateMutex.unlock(); 
          if(followMode==followModeMoveBaseGoal)
          {
            move_base_msgs::MoveBaseActionGoal goal_reset = GenerateGoal(img_secs, img_nsecs, odom_x, odom_y, odom_yaw);
            moveBaseGoalPublisher.publish(goal_reset);
            ros::spinOnce();
          }
          else
          {
            geometry_msgs::Twist twist;
            twistPublisher.publish(twist);  
            ros::spinOnce();
          }
          continue;
        }

        if(followMode==followModeMoveBaseGoal)
        {
          int move = 0;
          nh->getParam("/samsung_stz_main/move", move);
          if(move==1)
          {
            int da_sign = (delta_angle>0)?(1):(-1);
            if(fabs(delta_angle)>M_PI/18)
            {
              //ЗАметка: Если объект в задней полусфере, надо сначала отдавать команды только на поворот, а когда будет острый угол хотя бы
              //тогда уже и по положению управлять
              //std::cout<<"ROTAAATE!!!\n";
              geometry_msgs::Twist twist;
              twist.angular.z = da_sign*std::min<double>(max_angular_speed, fabs(delta_angle))*angular_speed_multiplier;
              twistPublisher.publish(twist);  
              ros::spinOnce();
            }  
            else
            {
              moveBaseGoalPublisher.publish(goal);
              ros::spinOnce();
            }
          }
        }
        else
        {

          //ЗАметка: Если объект в задней полусфере, надо сначала отдавать команды только на поворот, а когда будет острый угол хотя бы
          //тогда уже и по положению управлять
          //double delta_angle = dest_angle-odom_yaw;
          //delta_angle+=(delta_angle>M_PI) ? -M_PI*2 : (delta_angle<-M_PI) ? 2*M_PI : 0;
          //state_line<<"odom_yaw="<<odom_yaw<<" dest_angle="<<dest_angle<<" "<<" da="<<delta_angle<<" "; 
          int da_sign = (delta_angle>0)?(1):(-1);
          double angular_vel = 0;
          double linear_vel = 0;
          if(fabs(delta_angle)>M_PI/3)
          {
            //ЗАметка: Если объект в задней полусфере, надо сначала отдавать команды только на поворот, а когда будет острый угол хотя бы
            //тогда уже и по положению управлять
            linear_vel = 0;
            angular_vel = da_sign*max_angular_speed;

          }  
          else
          {
            if(fabs(delta_angle)>max_angular_speed)
            {
              angular_vel = da_sign*max_angular_speed;
            }
            else
            {
              angular_vel = da_sign*fabs(delta_angle);
            }
          } 

          double delta_r = fabs(sqrt(delta_x*delta_x + delta_y*delta_y));
          if(fabs(delta_r)>max_linear_speed)
          {
            linear_vel = max_linear_speed;
          }
          else
          {
            linear_vel = fabs(delta_r);
          }

          angular_vel*=angular_speed_multiplier;
          linear_vel*=linear_speed_multiplier;
          state_line<<" ang_vel = "<<angular_vel<<" lin_vel = "<<linear_vel<<" ";
          static int prev_move=0;
          int move = 0;
          nh->getParam("/samsung_stz_main/move", move);
          if(move==1)
          {
            geometry_msgs::Twist twist;
            twist.linear.x = linear_vel;
            twist.angular.z = angular_vel;
            twistPublisher.publish(twist);  
            ros::spinOnce();
          }
          if(move == 0 && prev_move ==1)
          {
            geometry_msgs::Twist twist;
            twistPublisher.publish(twist);  
            ros::spinOnce();
          }
          prev_move=move;
          
        }
        ros::spinOnce();
      }
    }
    else
    {
      state_line<<"State: ERROR ";
    }
    /*else if(state==GATHER_LOST_CUBE)
    {
      //If distance from desired object>threshold = move to desired object
      //if object reached but cube not touched move forward for x centimeters, and send message @cube lost@
      //If we touch cube then we switch state in grapple callback, and newer reach destination, NOTE IT
      state_line<<"State: gather lost in front, ";
  
      //Мы доехали, сменить состояние
      if(grapple_hold_cube)
      {
        state_line<<"gathered successfully ";
        currentStateMutex.lock();
        current_state = STATE_SEARCH;
        currentStateMutex.unlock(); 
        commandPublisher.publish(msg_box_taken);
        ros::spinOnce();
        geometry_msgs::Twist twist;
        twistPublisher.publish(twist);  
        ros::spinOnce();
      }

      //В общем, по плану это работает так
      //Приходим мы сюда если gathering_goal и только если задан
      //В режиме движения по стеку мы просто оцениваем разницу, и если она меньше порогов, то стопаем и говорим, что потеряли

      double dest_x = gathering_goal.goal.target_pose.pose.position.x;//  destination_object[4];
      double dest_y = gathering_goal.goal.target_pose.pose.position.y;
      double delta_x = dest_x - odom_x;
      double delta_y = dest_y - odom_y;
      double dest_angle = atan2(delta_y, delta_x);
      double delta_angle = dest_angle-odom_yaw;
      delta_angle+=(delta_angle>M_PI) ? -M_PI*2 : (delta_angle<-M_PI) ? 2*M_PI : 0;       

      geometry_msgs::PoseStamped gp;
      gp.header = gathering_goal.goal.target_pose.header;
      gp.pose = gathering_goal.goal.target_pose.pose;
      debugGoalPosePublisher.publish(gp);
      ros::spinOnce();


      if(fabs(delta_x)<gathering_delta_pos_allowed && fabs(delta_y)<gathering_delta_pos_allowed && fabs(delta_angle)<gathering_delta_angle_allowed)
      {
        //Считаем что прибыли в точку назначения, раз куб не захватили, то его и нет
        state_line<<" finish move, not found, end ";
        commandPublisher.publish(msg_box_not_taken);
        ros::spinOnce();
        currentStateMutex.lock();
        current_state = STATE_SEARCH;
        currentStateMutex.unlock();
        geometry_msgs::Twist twist;
        twistPublisher.publish(twist);  
        ros::spinOnce();
        continue;
      }

      if(followMode==followModeControlByCoord)
      { 
          //ЗАметка: Если объект в задней полусфере, надо сначала отдавать команды только на поворот, а когда будет острый угол хотя бы
          //тогда уже и по положению управлять
          //double delta_angle = dest_angle-odom_yaw;
          //delta_angle+=(delta_angle>M_PI) ? -M_PI*2 : (delta_angle<-M_PI) ? 2*M_PI : 0;
          //state_line<<"odom_yaw="<<odom_yaw<<" dest_angle="<<dest_angle<<" "<<" da="<<delta_angle<<" "; 
          int da_sign = (delta_angle>0)?(1):(-1);
          double angular_vel = 0;
          double linear_vel = 0;
          if(fabs(delta_angle)>M_PI/3)
          {
            //ЗАметка: Если объект в задней полусфере, надо сначала отдавать команды только на поворот, а когда будет острый угол хотя бы
            //тогда уже и по положению управлять
            linear_vel = 0;
            angular_vel = da_sign*max_angular_speed;
          }  
          else
          {
            if(fabs(delta_angle)>max_angular_speed)
            {
              angular_vel = da_sign*max_angular_speed;
            }
            else
            {
              angular_vel = da_sign*min_angular_speed,fabs(delta_angle);
            }
          } 

          double delta_r = fabs(sqrt(delta_x*delta_x + delta_y*delta_y));
          if(fabs(delta_r)>max_linear_speed)
          {
            linear_vel = max_linear_speed;
          }
          else
          {
            linear_vel = fabs(delta_r);
          }

          angular_vel*=angular_speed_multiplier;
          linear_vel*=linear_speed_multiplier;
          state_line<<" ang_vel = "<<angular_vel<<" lin_vel = "<<linear_vel<<" ";
          static int prev_move=0;
          int move = 0;
          nh->getParam("/samsung_stz_main/move", move);
          if(move==1)
          {
            geometry_msgs::Twist twist;
            twist.linear.x = linear_vel;
            twist.angular.z = angular_vel;
            twistPublisher.publish(twist);  
            ros::spinOnce();
          }
          if(move == 0 && prev_move == 1)
          {
            geometry_msgs::Twist twist;
            twistPublisher.publish(twist);  
            ros::spinOnce();
          }
          prev_move=move;     
      }
      else
      {
        //Задали цель и движемся к ней, тут управлять не надо, наверное?
      }  
      
      //static int counter = 0;
      //counter+=1;
      //if(counter>1000)
      //{
      //  counter=0;
      //  commandPublisher.publish(msg_box_not_taken);
      //  currentStateMutex.lock();
      //  current_state = STATE_SEARCH;
      //  currentStateMutex.unlock();
      //}
      
    }*/
    

    visualization_msgs::MarkerArray cubes_clear;
    for(int i=detections.size()+1; i<10; i++)
    {
        visualization_msgs::Marker cb;
        cb.id = i;
        cb.action=3;
        cubes_clear.markers.push_back(cb);
    }
    cubesPublisher.publish(cubes_clear);
    ros::spinOnce();
    cubesPublisher.publish(cubes);
    ros::spinOnce();


    std::cout<<state_line.str().c_str()<<"\n";
    //cv::imshow("view", debug_img);
    //cv::waitKey(10);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", debug_img).toImageMsg();
    debugImagePublisher.publish(msg);
    ros::spinOnce();
  }
  cv::destroyWindow("view");
  return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
move_base_msgs::MoveBaseActionGoal GenerateGoal(int sec, int nsec, double dest_x, double dest_y, double dest_angle)
{
  //Формируем цель
  move_base_msgs::MoveBaseActionGoal goal;
  goal.header.frame_id = "odom";
  goal.header.stamp.sec = sec;
  goal.header.stamp.nsec = nsec;
  goal.goal_id.stamp.sec = sec;
  goal.goal_id.stamp.nsec = nsec;
  goal.goal.target_pose.header.stamp.sec = sec;
  goal.goal.target_pose.header.stamp.nsec = nsec;
  goal.goal.target_pose.header.frame_id = "odom";
  goal.goal.target_pose.pose.position.z=0;
  goal.goal.target_pose.pose.position.x = dest_x;
  goal.goal.target_pose.pose.position.y = dest_y;
  tf::Quaternion qq = tf::createQuaternionFromYaw(dest_angle);
  tf::quaternionTFToMsg(qq,goal.goal.target_pose.pose.orientation);
  return goal;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
visualization_msgs::Marker GenerateMarker( int secs, int nsecs, int id, double x, double y, double scale, float r, float g, float b, float a)
{
  visualization_msgs::Marker cb;
  cb.header.frame_id = "odom";
  cb.header.stamp.sec = secs;
  cb.header.stamp.nsec = nsecs;
  cb.ns = "cube";
  cb.id = id;
  cb.type = 1;//CUBE
  cb.pose.position.x = x;
  cb.pose.position.y = y;
  cb.scale.x = scale/2.0;
  cb.scale.y = scale/2.0;
  cb.scale.z = scale/2.0;
  cb.color.r = r;
  cb.color.g = g;
  cb.color.b = b;
  cb.color.a = a;
  return cb;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Проинициализировать параметры и накидать их в список, если уже не заданы (скажем, через *.launch-файл)
void initParams(ros::NodeHandle* nh_p)
{
    if(!nh_p->hasParam("min_y_position"))
        nh_p->setParam("min_y_position",min_y_position);
    if(!nh_p->hasParam("minimal_conf"))
        nh_p->setParam("minimal_conf",minimal_conf);

    if(!nh_p->hasParam("max_angular_speed"))
        nh_p->setParam("max_angular_speed",max_angular_speed);
    if(!nh_p->hasParam("min_angular_speed"))
        nh_p->setParam("min_angular_speed",min_angular_speed);
    if(!nh_p->hasParam("angular_speed_multiplier"))
        nh_p->setParam("angular_speed_multiplier",angular_speed_multiplier);
    if(!nh_p->hasParam("max_linear_speed"))
        nh_p->setParam("max_linear_speed",max_linear_speed);
    if(!nh_p->hasParam("min_linear_speed"))
        nh_p->setParam("min_linear_speed",min_linear_speed);
    if(!nh_p->hasParam("linear_speed_multiplier"))
        nh_p->setParam("linear_speed_multiplier",linear_speed_multiplier);

    if(!nh_p->hasParam("min_trajectory_len"))
        nh_p->setParam("min_trajectory_len",minimalTrajectoryLen);

    if(!nh_p->hasParam("pixel_displacement_allowed"))
        nh_p->setParam("pixel_displacement_allowed",pixel_displacement_allowed);
    if(!nh_p->hasParam("metric_displacement_allowed"))
        nh_p->setParam("metric_displacement_allowed",metric_displacement_allowed);

    if(!nh_p->hasParam("vertical_desired_pos_rel"))
        nh_p->setParam("vertical_desired_pos_rel",vertical_desired_pos_rel);
    if(!nh_p->hasParam("maximum_lost_frames"))
        nh_p->setParam("maximum_lost_frames",maximum_lost_frames);
    if(!nh_p->hasParam("gathering_area_relative_width"))
        nh_p->setParam("gathering_area_relative_width",gathering_area_relative_width);

    if(!nh_p->hasParam("follow_meter_length"))
        nh_p->setParam("follow_meter_length",follow_meter_length);
    if(!nh_p->hasParam("gathering_delta_pos_allowed"))
        nh_p->setParam("gathering_delta_pos_allowed",gathering_delta_pos_allowed);
    if(!nh_p->hasParam("gathering_delta_angle_allowed"))
        nh_p->setParam("gathering_delta_angle_allowed",gathering_delta_angle_allowed);

    if(!nh_p->hasParam("follow_mode"))
        nh_p->setParam("follow_mode",followMode);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Прочитать значения параметров
void readParams(ros::NodeHandle* nh_p)
{
    if(nh_p->hasParam("min_y_position"))
        nh_p->getParam("min_y_position",min_y_position);
    if(nh_p->hasParam("minimal_conf"))
        nh_p->getParam("minimal_conf",minimal_conf);

    if(nh_p->hasParam("max_angular_speed"))
        nh_p->getParam("max_angular_speed",max_angular_speed);
    if(nh_p->hasParam("min_angular_speed"))
        nh_p->getParam("min_angular_speed",min_angular_speed);
    if(nh_p->hasParam("angular_speed_multiplier"))
        nh_p->getParam("angular_speed_multiplier",angular_speed_multiplier);
    if(nh_p->hasParam("max_linear_speed"))
        nh_p->getParam("max_linear_speed",max_linear_speed);
    if(nh_p->hasParam("min_linear_speed"))
        nh_p->getParam("min_linear_speed",min_linear_speed);
    if(nh_p->hasParam("linear_speed_multiplier"))
        nh_p->getParam("linear_speed_multiplier",linear_speed_multiplier);

    if(nh_p->hasParam("min_trajectory_len"))
        nh_p->getParam("min_trajectory_len",minimalTrajectoryLen);

    if(nh_p->hasParam("pixel_displacement_allowed"))
        nh_p->getParam("pixel_displacement_allowed",pixel_displacement_allowed);
    if(nh_p->hasParam("metric_displacement_allowed"))
        nh_p->getParam("metric_displacement_allowed",metric_displacement_allowed);

    if(nh_p->hasParam("vertical_desired_pos_rel"))
        nh_p->getParam("vertical_desired_pos_rel",vertical_desired_pos_rel);
    if(nh_p->hasParam("maximum_lost_frames"))
        nh_p->getParam("maximum_lost_frames",maximum_lost_frames);
    if(nh_p->hasParam("metric_displacement_allowed"))
        nh_p->getParam("metric_displacement_allowed",metric_displacement_allowed);

    if(nh_p->hasParam("follow_meter_length"))
        nh_p->getParam("follow_meter_length",follow_meter_length);
    if(nh_p->hasParam("gathering_delta_pos_allowed"))
        nh_p->getParam("gathering_delta_pos_allowed",gathering_delta_pos_allowed);
    if(nh_p->hasParam("gathering_delta_angle_allowed"))
        nh_p->getParam("gathering_delta_angle_allowed",gathering_delta_angle_allowed);

    if(nh_p->hasParam("follow_mode"))
        nh_p->getParam("follow_mode",followMode);

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  try
  {
    std::lock_guard<std::mutex> img_guard(imageMutex);
    currentImage = cv::imdecode(cv::Mat(msg->data),1);
    img_sec = msg->header.stamp.sec;
    img_nsec = msg->header.stamp.nsec;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert to image!");
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void calcBoxPosition(double u, double v, double mod, double &X, double&Z, double &angle){
	double cam_h, cam_angle, cx, cy, fx, fy, xcam, ycam, floor_angle;
	//camera position above ground, m
	cam_h = 0.07;
	//camera pitch angle, rad
	cam_angle = 0.075049;
	// camera intrinsics
  //Делим все на 2.5 так как калибровку мы выполняли на 800х600, а работаем на 320х240, ровно в 2.5 раза меньше!
  cx = (3.97842e+02)/mod;//2.5f;
  cy = (3.30757e+02)/mod;//2.5f;
  fx = (6.53585e+02)/mod;//2.5f;
  fy = (6.53487e+02)/mod;//2.5f;

  xcam = (u - cx)/fx;
  ycam = (v - cy)/fy;

  floor_angle = atan(ycam) - cam_angle;

  Z = cam_h/tan(floor_angle);
  X = Z * xcam;
  angle = atan(xcam);

  return;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void detectorCallback(const object_detection_msgs::DetectorResultConstPtr& msg)
{
    try
    {
        std::lock_guard<std::mutex> det_guard(detectionDataMutex);
        det_sec = msg->header.stamp.sec;
        det_nsec = msg->header.stamp.nsec;
        detectionRobotX = msg->x;
        detectionRobotY = msg->y;
        detectionRobotYaw = msg->angleZ;
        //std::cout<<"detectionRobotX = "<<detectionRobotX<<" detectionRobotY = "<<detectionRobotY<<" detectionRobotYaw = "<<detectionRobotYaw<<"\n";
        receivedRects.clear();
        for(int i=0; i<msg->res.size(); i++)
        {
            std::vector<float> r;
            r.resize(5, 0.0f);
            r[0] = msg->res[i].x_min;
            r[1] = msg->res[i].y_min;
            r[2] = msg->res[i].x_max;
            r[3] = msg->res[i].y_max;
            r[4] = msg->res[i].conf;
            receivedRects.push_back(r);
        }
    }
    catch (...)
    {
        ROS_ERROR("Exception occured during detector result receive");
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void OdomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
	std::lock_guard<std::mutex> odom_guard(odometryDataMutex);
	odometryX = msg->pose.pose.position.x;
	odometryY = msg->pose.pose.position.y;

	tf::Quaternion qq;
	tf::quaternionMsgToTF(msg->pose.pose.orientation, qq);
	odometryYaw = tf::getYaw(qq);
  odometryTimeStamp = msg->header.stamp.sec+static_cast<double>(msg->header.stamp.nsec)/1000000000;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CommandCallback(const std_msgs::String::ConstPtr& msg)
{
  ros::Publisher tp = nh->advertise<geometry_msgs::Twist>("/state_machine/cmd_vel", 100);
  std::lock_guard<std::mutex> cmd_guard(currentStateMutex);
  if(msg->data==std::string("GATHER_CUBE") && current_state == STATE_SEARCH)
  {
    current_state = FOLLOW_CUBE;
  }

  if(msg->data==std::string("STOP_GATHER"))
  {
    geometry_msgs::Twist message;
    message.linear.x = 0;
    message.angular.z = 0;
    tp.publish(message);
    ros::spinOnce();
    current_state = STATE_SEARCH;
  }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*//Прием данных от захвата
//STATE MACHINE
//Search for cubes and send the message as confident cube found
const int STATE_SEARCH = 0;
//Here we follow the cube using its pixel or metric coords, if cube is lost or no detection we still move to coords and hope to find new one in almost same place
const int FOLLOW_CUBE = 1;
//Here we move forward until reach cube or pass desired length
const int GATHER_LOST_CUBE = 2;
std::mutex currentStateMutex;
int current_state = 0;
*/
void grappleCallback(const std_msgs::Bool::ConstPtr& msg)
{
  grappleStateMutex.lock();
  grappleHoldCube = msg->data;
  grappleStateMutex.unlock();
}
