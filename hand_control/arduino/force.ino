
#include <ros.h>
#include <std_msgs/Float32MultiArray.h>

float values_flex[2] = {0.0, 0.0};
float values_force[2] = {0.0, 0.0};
float values[4] = {0.0, 0.0, 0.0, 0.0};
float ref[4] = {0.0, 0.0, 0.0, 0.0};

int n_sensors = 4;
int sensor_pin[4] = {3, 4, 5, 6} ;

ros::NodeHandle  nh;
unsigned long range_timer;
std_msgs::Float32MultiArray temp;

ros::Publisher pub_force("/gripper/force", &temp);
ros::Publisher pub_flex("/gripper/flex", &temp);


void setup()
{  
  nh.getHardware()->setBaud(9600);
  nh.initNode();
  temp.data_length = n_sensors / 2;
  nh.advertise(pub_force);
  nh.advertise(pub_flex);
}

void loop()
{
  nh.spinOnce();
  for (int i = 0; i < n_sensors; i++) {
    values[i] = (float)analogRead(sensor_pin[i]);
    if (millis() > 5000 & ref[i] == 0.0) ref[i] = values[i];
    values[i] = values[i] - ref[i];
    delay(10);
  }
  if ( (millis() - range_timer) > 50) {
    values_flex[0] = values[2];
    values_flex[1] = values[3];
    values_force[0] = values[0];
    values_force[1] = values[1];
    temp.data = values_force;
    pub_force.publish(&temp);
    delay(10);
    temp.data = values_flex;
    pub_flex.publish(&temp);
    delay(10);
    range_timer =  millis() + 50;
  }
}


//using openhand_node::ReadSensors;
//void callback(const Float32MultiArray::Request & req, Float32MultiArray::Response & res){
//
//    for (int i = 0; i < n_sensors; i++) {
//      values[i] = (float)analogRead(sensor_pin);
//      delay(100);
//      }
//    res.force = values;
//    res.flex = values;
//    res.force_length = n_sensors;
//    res.flex_length = n_sensors;
//}
//ros::ServiceServer<Float32MultiArray::Request, Float32MultiArray::Response> server("/openhand_node/force_flex",&callback);
