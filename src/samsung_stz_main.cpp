#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <mutex>
#include <list>
#include <map>

#include <ros/ros.h>
#include "ros/package.h"
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/Twist.h"
#include "std_msgs/String.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_datatypes.h>

//#include <sensor_msgs/PointCloud2.h>
//#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
//#include "pcl_ros/point_cloud.h"
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl_conversions/pcl_conversions.h>

//#include <sensor_msgs/Image.h>

#include "object_detection_msgs/DetectorResult.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
//#include <opencv2/tracking.hpp>
//#include <opencv2/core/ocl.hpp>





/////////////////////////////////////////////////////////////////////////
//TODO: Вынести параметры в rosparam
//TODO: Задание параметров из лаунч-файла
//Процент высоты картинки выше которого мы объекты не рассматриваем как кубы
float min_y_position = 0.6f;
//Минимальная уверенность детектора ниже которой объекты не рассматриваются
float minimal_conf = 0.6f;
//Минимальная уверенность трекера котрую пока не посчитать :(
//const float tracker_min_conf = 0.3f;
//Максимально допустимое изменение размера куба при сопровождении между кадрами
//abs(old_w-new_w)/old_w<... and abs(old_h-new_h)/old_h<...
float maximal_cube_size_change_allowed = 0.05f;
//Минимально допустимое смещение (pixels) при котором сопровождение считается потерянным
int pixel_displacement_allowed = 20;
int minimalTrajectoryLen = 2; //То есть два кадра подряд надо найти объект рядом чтобы писать сообщения
float metric_displacement_allowed = 0.2;//Can be mistaken for 20
bool useTracker = false;
/////////////////////////////////////////////////////////////////////////
//Максимальная угловая скорость платформы, ограничение в целях безопасности
float max_angular_speed = 0.5f;
float min_angular_speed = 0.05f;
//Максимальная линейная скорость платформы, ограничение в целях безопасности
float max_linear_speed = 0.8f;
float min_linear_speed = 0.05f;


//TRACKING PARAMETERS
//How to determine that curren cube is the same as last time?
//Multiplier to determine how much we can be moved from current position in % of prev size
float new_cube_percent_displacement = 1.5;
// HOw much frames in the row we should detect cube in near positionn to decide it is confident?
float allowed_trajectory_length = 2;

//Множитель для относительного смещения объекта от центра изображения по вертикали (угловая скорость)
float horizontal_displacement_multiplier = 0.5;
//Множитель для относительного смещения объекта от минимального положения по горизонтали (линейная скорость)
float vertical_displacement_multiplier = 1.0/3.0;
//Минмальное положение по вертикали ("желаемая" позиция куба по Y)
float vertical_desired_pos_rel = 0.98;
//Ho much time we should move to earlier detected object until say we lost it
//!!DesiredObject remain in old position, where earlier it was confidently deteted
int maximum_lost_frames = 50;
/////////////////////////////////////////////////////////////////////////
void initParams(ros::NodeHandle* nh_p);
void readParams(ros::NodeHandle* nh_p);
/////////////////////////////////////////////////////////////////////////
//Служебная функция для расчета положения куба в пространстве
void calcBoxPosition(double u, double v, double mod, double &X, double&Z, double &angle);
//Коллбэк на прием картинок
void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg);
//Коллбэк на прием рамок кубов
void detectorCallback(const object_detection_msgs::DetectorResultConstPtr& msg);
//Прием данных от одометрии
void OdomCallback(const nav_msgs::Odometry::ConstPtr& msg);
//Прием данных командного топика
void CommandCallback(const std_msgs::String::ConstPtr& msg);

//Текущее изображение для получения из коллбэка через мьютекс
cv::Mat currentImage;
std::mutex imageMutex;
//Время получения картинки (по времени ПК робота!)
int img_sec;
int img_nsec;

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
float delta_img_det=0.0f;
float delta_img_det_prev=0.0f;

//ODOMETRY
std::mutex odometryDataMutex;
float odometryX = 0.0f;
float odometryY = 0.0f;
float odometryYaw = 0.0f;
double odometryTimeStamp=0.0;

//Служебные РОСовские представители узла
ros::NodeHandle* nh;
ros::NodeHandle* nh_p;

//Тип трекера
//Хорошие - неплохо 1, кое-как 2, оч хорошо 4 и ужасно остальные
std::string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE"};
std::string trackerType = trackerTypes[4];

//Указатель на трекер OpenCV
cv::Ptr<cv::Tracker> tracker;
//Уверенность (заготовка уверенности) которой хз где взять
//float trackerConfidence = 0.0f;
//Текущая рамка и с прошлого кадра, используются для детекции резких срывов сопровождения
//cv::Rect2d tracker_bbox(0, 0, 0, 0);
//cv::Rect2d tracker_bbox_prev(0,0,0,0);

std::vector<double> previousDesiredObjectPosition;
std::vector<double> currentDesiredObjectPosition;
int desiredObjectTrajectoryLength=0;

//Флаг работы трекера, если сброшен в случае срыва или еще чего, трекер будет удален, затем создан и проинициализирован вновь
//bool trackerInitialized = false;

// Переменная состояния для куба - обнаружен / необнаружен
//bool tracking_ok = false;
//bool tracker_ok_old = false;

//bool objectFound = false;

//STATES HERE
//Search for cubes and send the message as confident cube found
const int STATE_SEARCH = 0;
//Here we follow the cube using its pixel or metric coords, if cube is lost or no detection we still move to coords and hope to find new one in almost same place
const int FOLLOW_CUBE = 1;
//Here we move forward until reach cube or pass desired length
const int GATHER_LOST_CUBE = 2;
std::mutex currentStateMutex;
int current_state = 0;

int main( int argc, char** argv )
{
    std_msgs::String msg;// = std_msgs::String();
    msg.data = "test";
    //Инициализация узла
    ros::init(argc, argv, "samsung_stz_main");
    ros::start();
    //Создать службу управления нодой
    nh = new ros::NodeHandle();
    nh_p = new ros::NodeHandle("~");

    ROS_INFO("Ros started!");

    // Сообщения для командного топика
    std_msgs::String msg_box_detected = std_msgs::String();
    std_msgs::String msg_box_not_detected = std_msgs::String();
    msg_box_detected.data = std::string("BOX_DETECTED");
    msg_box_not_detected.data = std::string("BOX_NOT_DETECTED");
    // Предыдущее состояние параметра
    int navigate_via_bbox = 0;
    int navigate_via_bbox_prev = 0;
    //Создать паблишеров и сабскрайберов
    ros::Subscriber sub = nh->subscribe("/usb_cam_front/image_raw/compressed", 1, imageCallback);
    ros::Subscriber sub_detector_res = nh->subscribe("/samsung/BBoxes", 1, detectorCallback);
    ros::Publisher twistPublisher = nh->advertise<geometry_msgs::Twist>("/state_machine/cmd_vel", 100);
    ros::Publisher commandPublisher = nh->advertise<std_msgs::String>("/kursant_driver/command", 100);
    ros::Publisher cubesPublisher = nh->advertise<visualization_msgs::MarkerArray>("/samsung/cube_positions", 20);

    //nh->setParam("/samsung/navigate_via_bbox", 0);

    //Может быть полезно если решим что-то сохранять в файлы или загружать
    //std::string package_path = ros::package::getPath("tld_tracker_sams")+std::string("/lib/");
    //if(!package_path.empty())
    //  std::cout<<"Package path: "<<package_path.c_str()<<"\n";

    //Создать окно отображения информации об объектах и сопровождении
    cv::namedWindow("view");
    cv::startWindowThread();

    //Создать трекер TODO: Перетащить в отдельную функцию
    /*
    if (trackerType == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    if (trackerType == "MIL")
        tracker = cv::TrackerMIL::create();
    if (trackerType == "KCF")
        tracker = cv::TrackerKCF::create();
    if (trackerType == "TLD")
        tracker = cv::TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        tracker = cv::TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    if (trackerType == "MOSSE")
        tracker = cv::TrackerMOSSE::create();*/

  initParams(nh_p);

  bool detectedConfidentObject=false;
  bool desired_object_lost = false;
  bool new_object_found = false;

  while(ros::ok())
  {
    //Считать параметры на случай если они изменились
    readParams(nh_p);

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
    //Без картинки все остальное - бессмысленно
    if(!debug_img.data)
    {
        ros::spinOnce();
        continue;
    }

    //Отдельная копия - для отладочного отображения
    tracking_img = debug_img.clone();

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

    //!Отладочная инфа рисуется всегда
    //Исходя из заданных в параметрах отосительных координат рассчитать положение "линии горизонта" и
    //"целевой" линии куда мы ведем низ куба
    int minimal_y = int(min_y_position*debug_img.rows);
    //std::cout<<"Myp="<<min_y_position<<" my="<<minimal_y<<" rows="<<debug_img.rows<<"\n";
    int desired_y = int(vertical_desired_pos_rel*debug_img.rows);
    int desired_x = debug_img.cols/2;
    //Отрисовать отладочные линии - "цель" по вертикали, горизонтали
    cv::line(debug_img, cv::Point2f(0, minimal_y), cv::Point2f(debug_img.cols-1, minimal_y), cv::Scalar(150,0,0), 1);
    cv::line(debug_img, cv::Point2f(0, desired_y), cv::Point2f(debug_img.cols-1, desired_y), cv::Scalar(0,150,0), 1);
    cv::line(debug_img, cv::Point2f(desired_x, 0), cv::Point2f(desired_x, debug_img.rows-1), cv::Scalar(0,150,0), 1);

    //Расчет разницы по времени между детекцией и текущим кадром
    delta_img_det = image_tstamp - detector_tstamp;

    double mod = 800.0/static_cast<double>(debug_img.cols);

    visualization_msgs::MarkerArray cubes;
    std::vector<double> desired_rect;
    if(detection_received)
    {    
        //Расчет пространственных координат куба для каждого прямоугольника
        //Опубликовать расчетные позы кубов
        

        
        std::vector<std::vector<float> > coords;
        for(int i=0; i<detections.size(); i++)
        {
            double X1,Z1,angle1, X2, Z2, angle2;
            float xmin = detections[i][0];
            float ymin = detections[i][1];
            float xmax = detections[i][2];
            float ymax = detections[i][3];
            float conf = detections[i][4];


            calcBoxPosition(xmin, ymax, mod, X1, Z1, angle1);
            calcBoxPosition(xmax, ymax, mod, X2, Z2, angle2);

            //Дополнительно - усреднить между правым нижним и левым нижним
            double X = (X1+X2)/2;
            double Z = (Z1+Z2)/2;
            double A = (angle1+angle2)/2;
            std::vector<float> r;
            r.resize(5, 0.0f);
            r[0] = X;
            r[1] = Z;
            r[2] = A;
            //calculate map positions
            double map_x = det_x + Z*cos(det_yaw) + X*sin(det_yaw);
            double map_y = det_y + Z*sin(det_yaw) - X*cos(det_yaw);
            r[3] = map_x;
            r[4] = map_y;
            coords.push_back(r);
            visualization_msgs::Marker cb;
            cb.header.frame_id = "map";
            cb.header.stamp.sec = img_secs;
            cb.header.stamp.nsec = img_nsecs;
            cb.ns = "cube";
            cb.id = i;
            cb.type = 1;//CUBE
            cb.pose.position.x = map_x; 
            cb.pose.position.y = map_y;
            cb.scale.x = 0.1;
            cb.scale.y = 0.1;
            cb.scale.z = 0.1;
            cb.color.r = 0.5;
            cb.color.g = 0.0;
            cb.color.b = 0.1;
            cb.color.a = 0.8;
            cubes.markers.push_back(cb);
            /*pcl::PointXYZRGB pt;
            pt.x = map_x;
            pt.y = map_y;
            pt.z = 0;
            pt.r = 150;
            pt.g = 150;
            pt.b = 150;
            cloud->push_back(pt);*/
        }

        cubesPublisher.publish(cubes);
        ros::spinOnce();  


        //Выбор того, к которому будем ехать
        //Просто выбираем из обнаруженных самый подходящий
        //Если сопровождение окей, то ничего не произойдет, а если не окей то произойдетъ
        //повторная инициализация ближайшим кубиком
        //std::vector<std::vector<float> > good_cubes;
        //std::vector<float> desired_rect;
        //We know nothing about where we are now
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
            prev_y = previousDesiredObjectPosition[4];
            //prev_==previousDesiredObjectPosition[1]
        }
        
        for(int i=0; i<detections.size(); i++)
        {
            float xmin = detections[i][0];
            float ymin = detections[i][1];
            float xmax = detections[i][2];
            float ymax = detections[i][3];
            float conf = detections[i][4];

            float curr_u = (xmin+xmax)/2;
            float curr_v = (ymin+ymax)/2;
            float curr_a = coords[i][2];
            float curr_x = coords[i][3];
            float curr_y = coords[i][4];

            if(ymin>minimal_y && conf>minimal_conf)
            {
                //good_cubes.push_back(detections[i]);
                cv::rectangle(debug_img, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(255,0,0), 2);
                std::stringstream ss;
                ss.setf(std::ios::fixed);
                ss.precision(2);
                ss<<"X="<<coords[i][0]<<" Y="<<coords[i][1];
                //Тут печатаем на картинке пространственные координаты
                cv::putText(debug_img, ss.str(), cvPoint(xmin,ymin-10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);

                if(prev_obj)
                {
                    if((fabs(curr_u-prev_u)<=pixel_displacement_allowed && 
                        fabs(curr_v-prev_v)<=pixel_displacement_allowed) || 
                        (fabs(curr_x-prev_x)<=metric_displacement_allowed && 
                        fabs(curr_y-prev_y)<=metric_displacement_allowed))
                    {
                        currentDesiredObjectPosition.resize(7); 
                        currentDesiredObjectPosition[0] = xmin;
                        currentDesiredObjectPosition[1] = ymin;
                        currentDesiredObjectPosition[2] = xmax;
                        currentDesiredObjectPosition[3] = ymax;
                        currentDesiredObjectPosition[4] = curr_x;
                        currentDesiredObjectPosition[5] = curr_y;
                        currentDesiredObjectPosition[6] = curr_a;
                        desiredObjectTrajectoryLength+=1;
                        break;
                    }
                }
                
                if(ymax>low_y)
                {
                    desired_rect.resize(7, 0.0f);
                    desired_rect.resize(7); 
                    desired_rect[0] = xmin;
                    desired_rect[1] = ymin;
                    desired_rect[2] = xmax;
                    desired_rect[3] = ymax;
                    desired_rect[4] = curr_x;
                    desired_rect[5] = curr_y;
                    desired_rect[6] = curr_a;
                }
                //Here we choose the best detection
                /*if(ymax>low_y)
                {
                    low_y = ymax;
                    desired_rect.resize(4, 0.0f);
                    desired_rect[0] = xmin+1;
                    desired_rect[1] = ymin+1;
                    desired_rect[2] = xmax-1;
                    desired_rect[3] = ymax-1;
                }*/
            }
        }
        
        //Now we check if we
        if((currentDesiredObjectPosition.size()>0) && (desiredObjectTrajectoryLength>minimalTrajectoryLen))
        {
            //We gonna use this flag to send message that object found
            detectedConfidentObject=true;
        }
        else
        {
            detectedConfidentObject=false;
        }

        //We lost desired object, this flag will be used by states later
        desired_object_lost = currentDesiredObjectPosition.size()==0;

        
            
        //
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
    if(state == STATE_SEARCH)
    {
      //We found confident object, send message
      if(detectedConfidentObject)
      {
          commandPublisher.publish(msg_box_detected);
          previousDesiredObjectPosition = currentDesiredObjectPosition;
      }
      else
      {
        //We have lost object or had no-one, but now found a new one, switch to it then
        if(new_object_found)
        {
          previousDesiredObjectPosition = desired_rect;
          desiredObjectTrajectoryLength = 1;
        }
        else //We lost an object and have no new one, as we do not move to it, drop it
        {
          previousDesiredObjectPosition.clear();
        }
      }
      
    }
    else if(state==FOLLOW_CUBE)
    {
      //Here we designate an object to which we want to 
      std::vector<double> target_object;
      static int lost_counter=0;
      if(detectedConfidentObject)
      {
          previousDesiredObjectPosition = currentDesiredObjectPosition;
          target_object = currentDesiredObjectPosition;
          lost_counter = 0;
      }
      else
      {
          //if(/*we lost object in front of us, assume we moving to gather it*/)
          //{
            //switch here to GATHER_LOST_CUBE
          //}
          //else
          {
            lost_counter+=1;
            if(lost_counter>maximum_lost_frames)
            {
              //We successfully lost cube
              //For now, we just reject old detection and move to new one if able to
              if(new_object_found)
              {
                previousDesiredObjectPosition = desired_rect;
                desiredObjectTrajectoryLength = 1;
                //This is very risky, new object not confident!!!
                //target_object=previousDesiredObjectPosition;
              }
              else
              {
                previousDesiredObjectPosition.clear();
              }
            }
            else
            {
              target_object=previousDesiredObjectPosition;
            }

          }

          if(target_object.size()>0)
          {

          }
      }
    }
    else if(state==GATHER_LOST_CUBE)
    {
      //If distance from desired object>threshold = move to desired object
      //if object reached but cube not touched move forward for x centimeters, and send message @cube lost@
      //If we touch cube then we switch state in grapple callback, and newer reach destination, NOTE IT 

    }
    else
    {
      
    }
    
    

    std::cout<<state_line.str().c_str();  
    cv::imshow("view", debug_img);
    cv::waitKey(10);
    ros::spinOnce();
  }
  cv::destroyWindow("view");
  return 0;
}
/////////////////////////////////////////////////////////////////////////
//float min_y_position = 0.6f;
//Минимальная уверенность детектора ниже которой объекты не рассматриваются
//float minimal_conf = 0.6f;
//Минимальная уверенность трекера котрую пока не посчитать :(
//const float tracker_min_conf = 0.3f;
//Максимально допустимое изменение размера куба при сопровождении между кадрами
//abs(old_w-new_w)/old_w<... and abs(old_h-new_h)/old_h<...
//float maximal_cube_size_change_allowed = 0.05f;
//Минимально допустимое смещение (в размерах рамки) при котором сопровождение считается потерянным
//float maximal_displacement_allowed = 0.5f;
/////////////////////////////////////////////////////////////////////////
//Максимальная угловая скорость платформы, ограничение в целях безопасности
//float max_angular_speed = 0.5f;
//float min_angular_speed = 0.05f;
//Максимальная линейная скорость платформы, ограничение в целях безопасности
//float max_linear_speed = 0.8f;
//float min_linear_speed = 0.05f;
//Множитель для относительного смещения объекта от центра изображения по вертикали (угловая скорость)
//float horizontal_displacement_multiplier = 0.5;
//Множитель для относительного смещения объекта от минимального положения по горизонтали (линейная скорость)
//float vertical_displacement_multiplier = 1.0/3.0;
//Минмальное положение по вертикали ("желаемая" позиция куба по Y)
//float vertical_desired_pos_rel = 0.98;
/////////////////////////////////////////////////////////////////////////

//Проинициализировать параметры и накидать их в список, если уже не заданы (скажем, через *.launch-файл)
void initParams(ros::NodeHandle* nh_p)
{
    if(!nh_p->hasParam("min_y_position"))
        nh_p->setParam("min_y_position",min_y_position);

    if(!nh_p->hasParam("minimal_conf"))
        nh_p->setParam("minimal_conf",minimal_conf);

    //if(!nh_p->hasParam("tracker_min_conf"))
    //    nh_p->setParam("tracker_min_conf",tracker_min_conf);

    /*if(!nh_p->hasParam("maximal_cube_size_change_allowed"))
        nh_p->setParam("maximal_cube_size_change_allowed",maximal_cube_size_change_allowed);

    if(!nh_p->hasParam("maximal_displacement_allowed"))
        nh_p->setParam("maximal_displacement_allowed",maximal_displacement_allowed);*/

    if(!nh_p->hasParam("max_angular_speed"))
        nh_p->setParam("max_angular_speed",max_angular_speed);

    if(!nh_p->hasParam("min_angular_speed"))
        nh_p->setParam("min_angular_speed",min_angular_speed);

    if(!nh_p->hasParam("max_linear_speed"))
        nh_p->setParam("max_linear_speed",max_linear_speed);

    if(!nh_p->hasParam("min_linear_speed"))
        nh_p->setParam("min_linear_speed",min_linear_speed);

    //if(!nh_p->hasParam("horizontal_displacement_multiplier"))
    //    nh_p->setParam("horizontal_displacement_multiplier", horizontal_displacement_multiplier);

    //if(!nh_p->hasParam("vertical_displacement_multiplier"))
    //    nh_p->setParam("vertical_displacement_multiplier", vertical_displacement_multiplier);

    //if(!nh_p->hasParam("vertical_desired_pos_rel"))
    //    nh_p->setParam("vertical_desired_pos_rel",vertical_desired_pos_rel);
}

//Прочитать значения параметров
void readParams(ros::NodeHandle* nh_p)
{
    if(nh_p->hasParam("min_y_position"))
        nh_p->getParam("min_y_position",min_y_position);

    if(nh_p->hasParam("minimal_conf"))
        nh_p->getParam("minimal_conf",minimal_conf);

    //if(nh_p->hasParam("tracker_min_conf"))
    //    nh_p->getParam("tracker_min_conf",tracker_min_conf);

    /*if(nh_p->hasParam("maximal_cube_size_change_allowed"))
        nh_p->getParam("maximal_cube_size_change_allowed",maximal_cube_size_change_allowed);

    if(nh_p->hasParam("maximal_displacement_allowed"))
        nh_p->getParam("maximal_displacement_allowed",maximal_displacement_allowed);*/

    if(nh_p->hasParam("max_angular_speed"))
        nh_p->getParam("max_angular_speed",max_angular_speed);

    if(nh_p->hasParam("min_angular_speed"))
        nh_p->getParam("min_angular_speed",min_angular_speed);

    if(nh_p->hasParam("max_linear_speed"))
        nh_p->getParam("max_linear_speed",max_linear_speed);

    if(nh_p->hasParam("min_linear_speed"))
        nh_p->getParam("min_linear_speed",min_linear_speed);

    //if(nh_p->hasParam("horizontal_displacement_multiplier"))
    //    nh_p->getParam("horizontal_displacement_multiplier", horizontal_displacement_multiplier);

    //if(nh_p->hasParam("vertical_displacement_multiplier"))
    //    nh_p->getParam("vertical_displacement_multiplier", vertical_displacement_multiplier);

    //if(nh_p->hasParam("vertical_desired_pos_rel"))
    //    nh_p->getParam("vertical_desired_pos_rel",vertical_desired_pos_rel);
}

void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  try
  {
    std::lock_guard<std::mutex> img_guard(imageMutex);
    currentImage = cv::imdecode(cv::Mat(msg->data),1);//convert compressed image data to cv::Mat
    //cv::imshow("view", image);
    //std::cout<<"Received image: "<<msg->header.stamp.sec<<"."<<msg->header.stamp.nsec<<"\n";
    img_sec = msg->header.stamp.sec;
    img_nsec = msg->header.stamp.nsec;
    //cv::waitKey(10);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert to image!");
  }
}

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

