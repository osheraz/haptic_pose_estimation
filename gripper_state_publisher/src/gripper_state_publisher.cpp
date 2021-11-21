
#include "gripper_state_publisher.hpp"
#include "openhand_node/ReadServos.h"
#include "openhand_node/ReadLoad.h"
#include "openhand_node/ReadCurrent.h"
#include "openhand_node/ReadTemperature.h"
//#include "common_msgs_gl/GetDoubleArray.h"

GripperStatePublisher::GripperStatePublisher(double frequency):loop_rate_(frequency) {
srv_clnt_pos_ = n_.serviceClient<openhand_node::ReadServos>("/openhand_node/read_servos");
srv_clnt_load_ = n_.serviceClient<openhand_node::ReadLoad>("/openhand_node/read_load");
srv_clnt_current_ = n_.serviceClient<openhand_node::ReadCurrent>("/openhand_node/read_current");
srv_clnt_temp_ = n_.serviceClient<openhand_node::ReadTemperature>("/openhand_node/read_temperature");
pub_pos_ = n_.advertise<std_msgs::Float32MultiArray>("/gripper/pos",1);
pub_load_ = n_.advertise<std_msgs::Float32MultiArray>("/gripper/load",1);
//pub_curr_ = n_.advertise<std_msgs::Float32MultiArray>("/gripper/curr",1);
pub_temp_ = n_.advertise<std_msgs::Float32MultiArray>("/gripper/temperature",1);
std::cout << "[GripperStatePublisher] Let the Magic Begin!\n " ;

}

void GripperStatePublisher::spin(){
	int temp_counter = 900;
	while(ros::ok()){

		openhand_node::ReadServos srv_pos;
		openhand_node::ReadLoad srv_load;
		openhand_node::ReadTemperature srv_temp;
		openhand_node::ReadCurrent srv_current;

		srv_clnt_pos_.call(srv_pos);
		srv_clnt_load_.call(srv_load);
		srv_clnt_current_.call(srv_current);

		if (temp_counter >= 1000)
		{
			srv_clnt_temp_.call(srv_temp);
			temp_counter = 0;
			temp.data = srv_temp.response.temp;
		}
		++temp_counter;

		std_msgs::Float32MultiArray position;
		position.data = srv_pos.response.pos;

		std_msgs::Float32MultiArray load;
		load.data = srv_load.response.load;

		std_msgs::Float32MultiArray current;
		current.data = srv_current.response.curr;

		pub_pos_.publish(position);
		pub_load_.publish(load);
		//pub_curr_.publish(current);

		pub_temp_.publish(temp);
		ros::spinOnce();
		loop_rate_.sleep();
	}
	
}
