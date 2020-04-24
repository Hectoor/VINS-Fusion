/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"

//把body坐标系加速度传进来，目的是计算世界坐标系下水平放置时与初始IMU的旋转矩阵?
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();   //标准化，模为1
    Eigen::Vector3d ng2{0, 0, 1.0}; //世界坐标轴下水平放置时的加速度 z轴指向地心
    //返回的旋转矩阵是R*ng1 = ng2   通过加速度来计算两者的旋转
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    //得到yaw轴角角度
    double yaw = Utility::R2ypr(R0).x();
    //
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
