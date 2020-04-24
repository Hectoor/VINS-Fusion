/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"
//构造函数，该类在rosNodeTest里被实例化
Estimator::Estimator(): f_manager{Rs}
{
    //初始化，清除所有状态
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}
//清除状态
void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}
//设置参数
//主要的功能就是开启前端处理线程processMeasurements
void Estimator::setParameter()
{
    mProcess.lock();    // 锁mProcess主要用于processThread线程互斥安全
    for (int i = 0; i < NUM_OF_CAM; i++)//设置左右相机与IMU（或体坐标系）之间的变换矩阵
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);//以下设置图像因子需使用的sqrt_info（即信息矩阵的根号值），该设置假定图像特征点提取存在1.5个像素的误差，
                          //存在FOCAL_LENGTH,是因为图像因子残差评估在归一化相机坐标系下进行
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);// 设置相机内参，该参数主要用于特征点跟踪过程

    //若多线程flag打开且还未初始化，打开观测数据处理线程（前端线程）
    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);//新线程
    }
    mProcess.unlock();
}


// 改变传感器类型
void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    //更改传感器配置后，vio系统重启
    //此处重启与检测到failure后的重启一样，主要包括清除状态、设置参数两步
    if(restart)
    {
        clearState();
        setParameter();
    }
}
//把rosNodeTest.cpp中的image0、image1传进来
//@parameter t：image`s time
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    //1.把筛选好的图片读进来
    inputImageCnt++;
    // 图像的特征点,使用map存放，每个特征点为7*1的向量，存放相机id、位置（x,y,z）、特征点在当前图像的位置（u,v）、速度
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;
    //2.跟踪trackImage()
    if(_img1.empty())
        //单目情况
        featureFrame = featureTracker.trackImage(t, _img);
    else
        //双目情况
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    //3.rviz显示跟踪的图像  左视图
    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();  //这个是获得相机位姿的函数
        pubTrackImage(imgTrack, t); //发布topic出去 这样就能在rviz显示
        //printf("pub track_image");
    }
    //4.使用多线程
    if(MULTIPLE_THREAD)
    {
        //这个if是什么意思？
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            //make_pair就是把两者结合在一起
            //featureBuf就存放特征点
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    //如果没有使用多线程，就相当于只有sync_process这个线程
    else
    {
        //打开processMeasurements()函数进行处理
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }

}
//读取imu数据进来
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    //1.时间戳和imu数据结合传到Buf里面
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();
    //2.已经初始化后，就对IMU进行处理（预积分等），并发布最新的位姿给可视化界面?
    //后来发现貌似没关系的  代码冗余
    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        //IMU 中值积分预测
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        // 发布中值积分预测的状态，主要是p、v、q？
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

//输入特征
void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}

//将t0到t1之间的IMU数据存放在accVector和gyrVector向量中
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    //vector的.back()指的是向量的末尾元素,末尾元素要大于t1时刻
    if(t1 <= accBuf.back().first)
    {
        //把<t0时刻的数据都丢掉
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        //把t0～t1之间的都留下来
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        //再加一个大于t1的IMU帧
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

// IMU是否可用 兩個判別準則
// 1判斷topic中的buf是否为空
// 2.back()表示末尾元素，判別IMU最末尾元素時間戳小於當前圖像幀時間戳
bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

//处理观测值线程（一直在运行） 这是一个综合的处理函数（包含图像和IMU的处理） 很重要
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        //定义一个pair类型存放数据，feature.first是double类型，feature.second是map类型，是图片数据
        //first:时间戳，second:特征点ID，对应7维数据：归一化相机坐标系坐标（3维），去畸变图像坐标系坐标2维，特征点速度2维
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        //如果featureBuf里面有数据，说明有解析好的特征点，所以可以搞事情了(在inputImage里解析)
        if(!featureBuf.empty())
        {
            //取一帧图像的特征点出来 td是时间偏移
            feature = featureBuf.front();
            curTime = feature.first + td;//时间偏差补偿后的图像帧时间戳
            while(1)
            {
                //1.IMU开启，如果IMU有数据 跳出这个循环
                //2.IMU关闭，！USE_IMU恒为true，直接跳出循环
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                //使用IMU
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            //有了imu数据，可以进一步搞事情
            mBuf.lock();
            //首先把上一时刻图像帧和这一时刻图像帧之间的imu数据取出来放到accVector、gyrVector中
            //t0～t1数据再加上大于t1的一帧
            if(USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);
            //特征点不要了，pop掉了，因为已经存到feature里了，看上面
            featureBuf.pop();
            mBuf.unlock();
            //先处理IMU数据
            if(USE_IMU)
            {
                //看看有没初始化，没就去初始化
                if(!initFirstPoseFlag)           //位姿未初始化，则利用加速度初始化Rs[0]
                    initFirstIMUPose(accVector);    //基于重力，对准第一帧，即将初始姿态对准到重力加速度方向
                //加速度遍历搞事情 这个循环实际上是计算了t0时刻相机帧到t1时刻相机帧的积分
                //两时刻图像帧之间有几帧imu数据，经过下面的循环把这几帧IMU积分都加起来了
                //需注意IMU帧和图像帧是不同的
                //这里和vins-mono有点不同，没有用插值法插到图像帧中，而是直接用上一时刻的加、角速度做中值积分
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;// accVector.first指的是时间戳
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    //xxxVector.second指的是IMU数据  在这里进行预积分操作
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            //再处理图像帧数据，优化、滑窗等
            mProcess.lock();
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);
            //发布信息
            pubOdometry(*this, header);
            pubKeyPoses(*this, header);     //pose_graph没有用到
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);             //loop closing 需用 用到
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

//初始化IMU的pose，利用重力信息，初始化最开始状态中的旋转矩阵
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    //初始化flag置1
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    //取加速度平均值
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    //个人感觉就是矫正世界坐标系，本来是使用相机/imu第一帧来作为世界坐标系，但z轴不一定对的上
    // Rs[0]存放的是当前时刻的body系到世界坐标系旋转
    // 主要利用基于重力方向，得到的roll pitch信息，由于yaw信息通过重力方向，并不能恢复出来，因此减去yaw??
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    //x,y,z轴旋转的角度，转成旋转矩阵的表示方式
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

//初始化pose
void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

//处理IMU
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    //如果不是第一帧imu，就直接赋值acc_0，gyr_0
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    // 如果没有创建预积分对象，则创建，frame_count指的是slide windwos的帧数
    if (!pre_integrations[frame_count])
    {
        //Bas,Bgs指的是imu的Bias
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    //滑窗中图片的帧数,最大为WINDOW_SIZE
    if (frame_count != 0)
    {
        //进行预积分操作 对每一帧图像的数据进行预积分 注意这里是预计分量，也就是图像帧之间的多帧IMU预积分
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        //存储两帧图像帧之间的多帧IMU测量值到buf
        dt_buf[frame_count].push_back(dt);                                      //积分时间
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);    //加速度
        angular_velocity_buf[frame_count].push_back(angular_velocity);          //角速度

        //在这里进行中值积分，得到当前时刻的P、V、R！
        //中值积分，此处与预积分中的中值积分基本相似；
        //但此处的中值积分是以世界坐标系为基准，即更新的Rs、Ps、Vs在世界坐标系下表达 tzhang
        int j = frame_count;
        //上一时刻的《世界坐标轴》的加速度，Rs[j]为body系转到世界坐标
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        //中值角速度
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        //计算旋转（角速度×时间），Rs指的是当前时刻body系旋转到世界坐标系
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        //当前时刻的无偏加速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        //中值加速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        //当前时刻的位移
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        //当前时刻的速度
        Vs[j] += dt * un_acc;
    }
    //储存这一时刻的数据，方便下一帧来的时候做中值积分
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//处理一帧图像
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());   //特征点的数量
    //如果有视差，则说明是关键帧，则边缘化最老那个帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    //如果没视差（非关键帧），则边缘化第二帧
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;  //时间戳

    ImageFrame imageframe(image, header);//实例化一个ImageFrame对象 传入特征点和时间戳
    imageframe.pre_integration = tmp_pre_integration; //传入预积分对象
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    //如果外参不确定（相机和IMU的旋转）
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            //geyCorresponding()获得两帧之间的匹配点对
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 执行标定程序计算外参旋转 把角点corres和预积分数据传进去  位移呢？位移没有标定,不需要
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                //标定出来的旋转
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
    //初始化
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU) //单目+IMU版本的初始化
        {
            if (frame_count == WINDOW_SIZE)//滑窗满后才开始初始化
            {
                bool result = false;
                //当前图像时间戳与上一次初始化的时间戳间隔大于0.1秒、且外参存在初始值才进行初始化操作
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    //初始化
                    result = initialStructure();
                    initial_timestamp = header;
                }
                if(result)//初始化成功
                {
                    //优化、更新状态、滑窗一条龙服务
                    optimization();
                    //获取滑窗中最新帧时刻的状态，并在世界坐标系下进行中值积分；重要：初始化完成后，最新状态在inputIMU函数中发布
                    //但似乎没啥用
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
        //初始化失败也需要进行滑窗处理。若global sfm求解引起的失败，则一律移除最老的图像帧；
        //否则，根据addFeatureCheckParallax的判断，决定移除哪一帧
                    slideWindow();
            }
        }

        // stereo + IMU initilization  双目+IMU的初始化
        if(STEREO && USE_IMU)
        {
            //对来的帧进行三角化并且初始化PnP
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            //当帧数填满窗口满之后
            if (frame_count == WINDOW_SIZE)
            {
                //这一块好像没用
                // fix one
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                //给一个IMU积分得到的位置、旋转初始值给图像帧？
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                //优化陀螺仪偏差  qc0_bk+1 = qc0_bk x qbk_bk+1通过这个约束求bias
                solveGyroscopeBias(all_image_frame, Bgs);
                //对每一帧数据进行重新传播
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    //pre_integrations预积分项的个数等于滑窗
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization(); //优化
                updateLatestStates();//更新最新状态
                solver_flag = NON_LINEAR; // 初始化完成，更变flag
                slideWindow();  //滑动窗口算法

                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        //frame_count是滑动窗口算法的索引 滑窗未满
        //下一图像帧时刻的状态量用上一图像帧时刻的状态量进行初始化？
        //（PS：processIMU函数中，会对Ps、Vs、Rs用中值积分进行更新）
        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;  //图像加1
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    //初始化完成之后
    else
    {
        TicToc t_solve;
        //当不存在imu时，使用pnp方法进行位姿预测;存在imu时使用imu积分进行预测
        //预测就是当成是相机位姿的初值去优化？
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        //双目三角化双目图，单目三角化前后帧，对未初始化landmark初始化
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);//三角化
        optimization();         //局部优化 Local BA
        set<int> removeIndex;   //容器
        outliersRejection(removeIndex);  //外点排除
        f_manager.removeOutlier(removeIndex);   //特征点管理器移除外点
        //非多线程
        if (! MULTIPLE_THREAD)
        {
            //若路标点为外点，则对前端图像跟踪部分的信息进行剔除更新;主要包括prev_pts, ids， track_cnt
            featureTracker.removeOutliers(removeIndex);
            //预测路标点在下一时刻左图中的坐标，基于恒速模型
            predictPtsInNextFrame();
        }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        //错误检测，重启系统
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();//重新开vio
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
        //滑动窗口
        slideWindow();
        f_manager.removeFailures(); //丢掉跟踪失败的点？
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        //更新上一帧旋转和位置
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();//更新状态发给RVIZ
    }
}
//初始化结构？
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    { //TODO(tzhang):该作用域段主要用于检测IMU运动是否充分，但是目前代码中，运动不充分时并未进行额外处理，此处可改进；或者直接删除该段
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {//对除第一帧以外的所有图像帧数据进行遍历,计算加速度均值
            double dt = frame_it->second.pre_integration->sum_dt;
            //通过速度计算，上一图像帧时刻到当前图像帧时刻的平均加速度
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g; //平均加速度累积
        }
        Vector3d aver_g;
        //计算所有图像帧间的平均加速度均值（注意size-1，因为N个图像帧，只有N-1个平均加速度）
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {//对除第一帧以外的所有图像帧数据进行遍历,计算加速度方差
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)//对标准差进行判断
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];  //R_w_c  from camera frame to world frame. tzhang
    Vector3d T[frame_count + 1];  // t_w_c
    map<int, Vector3d> sfm_tracked_points;  //观测到的路标点的在世界坐标系的位置，索引为路标点的编号
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {//对所有路标点进行遍历
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        //对观测到路标点j的所有图像帧进行遍历
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //通过本质矩阵求取滑窗最后一帧（WINDOW_SIZE）到图像帧l的旋转和平移变换
    if (!relativePose(relative_R, relative_T, l))
    {//共视点大于20个、视差足够大，才进行求取
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;// 通过global sfm求取滑窗中的图像帧位姿，以及观测到的路标点的位置
    //只有frame_count == WINDOW_SIZE才会调用initialStructure，
    // 此时frame_count即为WINDOW_SIZE
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame 对所有图像帧处理，并得到imu与world之间的变换
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    //对所有图像帧进行遍历，i为滑窗图像帧index，frame_it为所有图像帧索引；
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {//滑窗图像帧是所有图像帧的子集,由于滑窗中可能通过MARGIN_SECOND_NEW，边缘化某些中间帧
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])//该图像帧在滑窗里面
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();//得到R_w_i
            frame_it->second.T = T[i];
            i++;
            continue;//若图像帧在滑窗中，直接利用上述global sfm的结果乘以R_c_i，得到imu到world的变换矩阵即R_w_i
        }
        // 时间戳比较，仅仅在所有图像的时间戳大于等于滑窗中图像帧时间戳时，才递增滑窗中图像时间戳
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        //由于后续进行PnP求解，需要R_c_w与t_c_w作为初值；且以滑窗中临近的图像帧位姿作为初值
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        //对图像帧观测到的所有路标点进行遍历
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first; //路标点的编号
            for (auto &i_p : id_pts.second)//TODO(tzhang):对左右相机进行遍历（PS：该循环放在if判断后更好）
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;//得到路标点在世界坐标系中的位置
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();//得到路标点在归一化相机坐标系中的位置
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if(pts_3_vector.size() < 6)//该图像帧不在滑窗中，且观测到的、已完成初始化的路标点数不足6个
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();  // 通过PnP求解得到R_w_c
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);  // 通过PnP求解得到t_w_c
        frame_it->second.R = R_pnp * RIC[0].transpose();  //得到R_w_i
        frame_it->second.T = T_pnp;  //t_w_i （PS：未考虑camera与imu之间的平移,由于尺度因子未知，此时不用考虑；在求解出尺度因子后，会考虑）
    }
    /* visualInitialAlign
    根据VIO课程第七讲:一共分为5步:
    1估计旋转外参. 2估计陀螺仪bias 3估计中立方向,速度.尺度初始值 4对重力加速度进一步优化 5将轨迹对其到世界坐标系 */
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}
//IMU与camera对准，计算更新了陀螺仪偏置bgs、重力向量g、尺度因子s、速度v
/* visualInitialAlign
根据VIO课程第七讲:一共分为5步:
1估计旋转外参. 2估计陀螺仪bias 3估计中立方向,速度.尺度初始值 4对重力加速度进一步优化 5将轨迹对其到世界坐标系 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {//Headers[i]存储图像帧时间戳，初始化过程中，仅边缘化老的图像帧，因此留在滑窗中的均为关键帧
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi; //t_w_b
        Rs[i] = Ri; //R_w_b
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);    //尺度因子
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        //得到陀螺仪bias新值，预积分重新传播；此处的预积分为Estimator类中的预积分；
        //图像帧中预积分的重新传播，在计算bg后已经完成
        //TODO(tzhang)：Estimator与图像帧中均维护预积分，重复，是否有必要？待优化
    }
    //利用尺度因子，对t_w_b的更新；注意此时用到了camera与imu之间的平移量
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    //更新Estimator中的速度，注意此时的速度值相对于世界坐标系；
    //而初始化过程中的速度，相对于对应的机体坐标系
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);//根据基于当前世界坐标系计算得到的重力方向与实际重力方向差异，计算当前世界坐标系的修正量；
    //注意：由于yaw不可观，修正量中剔除了yaw影响，也即仅将世界坐标系的z向与重力方向对齐
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;  //将世界坐标系与重力方向对齐，之前的世界坐标系Rs[0]根据图像帧定义得到，并未对准到重力方向
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();  //清除路标点状态，假定所有路标点逆深度均为估计；注意后续优化中路标点采用逆深度，而初始化过程中路标点采用三维坐标
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);  //基于SVD的路标点三角化，双目情形：利用左右图像信息； 非双目情形：利用前后帧图像信息

    return true;
}
//返回值：WINDOW_SIZE变换到l的旋转、平移；及图像帧index

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)  //遍历滑窗中的图像帧（除去最后一个图像帧WINDOW_SIZE）
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);  //获取图像帧i与最后一个图像帧（WINDOW_SIZE）共视路标点在各自左归一化相机坐标系的坐标
        if (corres.size() > 20)  //图像帧i与最后一个图像帧之间的共视点数目大于20个才进行后续处理
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;  //计算归一化相机坐标系下的视差和

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());  //归一化相机坐标系下的平均视差
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {  //上述的460表示焦距f（尽管并不太严谨，具体相机焦距并不一定是460），从而在图像坐标系下评估视差
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

//向量类型转成double类型，因为等下要优化，ceres优化器要用double类型才可以
/*可以看出来，这里面生成的优化变量由：
para_Pose（7维，相机位姿）、
para_SpeedBias（9维，相机速度、加速度偏置、角速度偏置）、
para_Ex_Pose（6维、相机IMU外参）、
para_Feature（1维，特征点深度）、
para_Td（1维，标定同步时间）
五部分组成，在后面进行边缘化操作时这些优化变量都是当做整体看待。*/
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);

        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}
// 检测不到点
bool Estimator::failureDetection()
{
    return false;  //TODO(tzhang):失败检测策略还可自己探索
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}
//局部Bundle Adjustment
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    //向量vector类型转double类型
    vector2double();

    //用ceres优化，定义一个问题
    ceres::Problem problem;
    //Huber核函数
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    //滑窗中的每一帧变量都要求解！
    /*######优化参数：q、p；v、Ba、Bg#######*/
    for (int i = 0; i < frame_count + 1; i++)
    {//IMU存在时，frame_count等于WINDOW_SIZE才会调用optimization()

        //ParameterBlock就是待求参数  SIZE_POSE= 7 ，位移+旋转（四元数）? x,y,z,qx,qy,qz,w
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        //IMU传感器速度、Bias也是要被估计的变量， vx,vy,vz,bgx,bgy,bgz,bax,bay,baz
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 没使用imu时,将窗口内第一帧的位姿固定
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    //外参也作为变量添加到ParameterBlock里
    /*######优化参数：imu与camera外参#######*/
    for (int i = 0; i < NUM_OF_CAM; i++)
    {//imu与camera外参
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            //保持参数块不变
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    //滑窗第一时刻相机与IMU时钟差作为优化变量
    /*######优化参数：imu与camera之间的time offset#######*/
    problem.AddParameterBlock(para_Td[0], 1);

    //如果不估计时钟，或速度小于0.2，则不估计时钟差
    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    //要添加残差了
    //边缘化残差
    /*******先验残差*******/
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor 残差函数
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        /* 通过提供参数块的向量来添加残差块。
        ResidualBlockId AddResidualBlock(
            CostFunction* cost_function,//损失函数
            LossFunction* loss_function,//核函数
            const std::vector<double*>& parameter_blocks); */
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    //IMU残差
    /*******预积分残差*******/
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)   //预积分残差，总数目为frame_count
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)//两图像帧之间时间sum_dt过长，不使用中间的预积分 tzhang
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);//残差函数
            //添加残差格式：残差因子，鲁棒核函数，优化变量（i时刻位姿，i时刻速度与偏置，i+1时刻位姿，i+1时刻速度与偏置）
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
           // para_SpeedBias[i], para_Pose[j], para_SpeedBias[j] 是需要优化的变量
        }
    }

    //视觉残差
    /*******重投影残差*******/
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)//遍历路标点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();  //路标点被观测的次数
        if (it_per_id.used_num < 4) //被观测次数太少了，直接跳过
            continue;

        ++feature_index;    //第一帧

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;   //??

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;  //用于计算估计值

        for (auto &it_per_frame : it_per_id.feature_per_frame) //遍历观测到路标点的图像帧
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;    //
                //左相机在i时刻和j时刻分别观测到路标点
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }
            //双目
            if(STEREO && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {   //左相机在i时刻、右相机在j时刻分别观测到路标点
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {   //左相机和右相机在i时刻分别观测到路标点
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }

            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());
    //配置一些信息
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;//狗腿算法
    options.max_num_iterations = NUM_ITERATIONS;     //最大迭代次数
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        //最大求解时间
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    //求解！
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    //求解完了再转回vector
    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)// 滑窗没满！！
        return;

    /*%%%%%滑窗满了，进行边缘化处理%%%%%%%*/
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) //将最老的图像帧数据边缘化； tzhang
    {
        //marginalization_info包含了本轮边缘化之后的先验信息
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // 先验部分，基于先验残差，边缘化滑窗中第0帧时刻的状态向量
        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                //丢弃滑窗中第0帧时刻的位姿、速度、偏置
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor  建立新的边缘化因子
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //imu 预积分部分，基于第0帧与第1帧之间的预积分残差，边缘化第0帧状态向量
        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); //边缘化 para_Pose[0], para_SpeedBias[0]
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //图像部分，基于与第0帧相关的图像残差，边缘化第一次观测的图像帧为第0帧的路标点和第0帧
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)   //对路标点的遍历
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                //对观测到路标点的图像帧的遍历
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        //左相机在i时刻、在j时刻分别观测到路标点
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            //左相机在i时刻、右相机在j时刻分别观测到路标点
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            //左相机和右相机在i时刻分别观测到路标点
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        // 上面通过调用 addResidualBlockInfo() 已经确定优化变量的数量、存储位置、长度以及待优化变量的数量以及存储位置，
        //-------------------------- 下面就需要调用 preMarginalize() 进行预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();    //边缘化操作，注意和滑窗slidewindow()是不同的，slidewindow()是删掉变量
        ROS_DEBUG("marginalization %f ms", t_margin.toc());
        //仅仅改变滑窗double部分地址映射，具体值的通过slideWindow和vector2double函数完成；记住边缘化仅仅改变A和b，不改变状态向量
        //由于第0帧观测到的路标点全被边缘化，即边缘化后保存的状态向量中没有路标点;因此addr_shift无需添加路标点
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)  //最老图像帧数据丢弃，从i=1开始遍历
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];  // i数据保存到1-1指向的地址，滑窗向前移动一格
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;//删除掉上一次的marg相关的内容
        last_marginalization_info = marginalization_info;//marg相关内容的递归
        last_marginalization_parameter_blocks = parameter_blocks;//优化变量的递归，这里面仅仅是指针

    }
    //次新图像边缘化
    else
    {//存在先验边缘化信息时才进行次新帧边缘化;否则仅仅通过slidewindow，丢弃次新帧
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;//记录需要丢弃的变量在last_marginalization_parameter_blocks中的索引
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    //TODO(tzhang):仅仅只边缘化WINDOW_SIZE - 1位姿变量， 对其特征点、图像数据不进行处理
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize(); //构建parameter_block_data
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            //仅仅改变滑窗double部分地址映射，具体值的更改在slideWindow和vector2double函数完成
            //由于边缘化次新帧，边缘化的状态向量仅为para_Pose[WINDOW_SIZE - 1];而保留的状态向量为在上一次边缘化得到的保留部分基础上、剔除para_Pose[WINDOW_SIZE - 1]的结果;
            //因此，边缘化次新帧得到的保留部分也未包含路标点，因此addr_shift无需添加路标点
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)  //WINDOW_SIZE - 1会被边缘化，不保存
                    continue;
                else if (i == WINDOW_SIZE)//WINDOW_SIZE数据保存到WINDOW_SIZE-1指向的地址
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else     //其余的保存地址不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];


            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;   //删除上一个边缘化信息
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}
// 滑动窗口
void Estimator::slideWindow()
{
    TicToc t_margin;
    //边缘化最老的帧
    if (marginalization_flag == MARGIN_OLD)
    {
        //把时间戳、旋转和平移取到变量
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        //窗口要满帧
        if (frame_count == WINDOW_SIZE)
        {
            //遍历滑窗 覆盖数据   *[WINDOW_SIZE]代表最新帧
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                //后面滑窗的时间戳、旋转和位移覆盖到前面
                //当i为WINDOW_SIZE-1时，不会出问题吗？不会，因为都是数组大小是WINDOWS_SIZE+1
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                //如果使用imu
                if(USE_IMU)
                {
                    // 把预积分量、积分时间、加速度、角速度也同时覆盖到前面
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);
                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);
                    // 速度、偏移量
                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            //question:让第滑窗+1个帧始终保持最新的数据？
            //XX[WINDOW_SIZE]存放的是最旧的数据，被边缘化的数据，见前面swap
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                //删除第滑窗+1个预积分量
                delete pre_integrations[WINDOW_SIZE];
                //创建第滑窗+1个预积分量
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
                //清空当前帧的数据
                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            //如果是初始化状态，清空数据
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);           //找到滑窗内最老一帧信息
                delete it_0->second.pre_integration;        //删除这一帧预积分数据
                all_image_frame.erase(all_image_frame.begin(), it_0);   //删除这一帧预积分数据？
            }
            //执行边缘化操作
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            //把第一帧覆盖到第二帧上去
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                //图像帧两帧的间隔中就有好几帧IMU数据，所以这里遍历指的是两帧图像之间那些IMU数据
                //比如两帧图像之间有三帧IMU数据，那么这里的size就是三
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    //question:为什么要重新计算呢？
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];
                //删除当前帧的预积分对象
                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            //边缘化第二帧
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;  //统计边缘化老帧的数量

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;

        R0 = back_R0 * ric[0];  //世界坐标系下最老帧的旋转
        R1 = Rs[0] * ric[0];    //?
        P0 = back_P0 + back_R0 * tic[0];    //世界坐标系下最老帧的平移
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}
//预测下一帧的pose 假设和上一帧速度一样
void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT; //当前帧、上一帧、下一帧的位姿
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    //基于恒速模型，预测下一时刻位姿 其实就是计算上一帧和当前帧的变换，这个变换再和当前帧相乘得到下一帧的预测
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)//特征点深度大于0，有效   //仅对已经初始化的路标点进行预测
        {
            int firstIndex = it_per_id.start_frame; //仅对已经初始化的路标点进行预测
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;//最后观测到该路标点的图像帧的index
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            //仅对观测次数不小于两次、且在最新图像帧中观测到的路标点进行预测
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth; //逆深度，在start_frame图像帧中表示
                //路标点在start_frame图像帧时刻，相机坐标系下的坐标
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                //路标点在世界坐标系下的坐标
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                //路标在在下一时刻（预测的）体坐标系下坐标
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                //路标在在下一时刻（预测的）相机坐标系下坐标
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                // 根据路标点编号，存储预测的坐标
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

//计算重投影误差，用于外点排除
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    //把第i帧的uv使用计算得到的P，V，R得到当前时刻的uv，相减  rici，tici是外参
    // 画图可以很好地看出来
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;//求出i帧空间点相对于世界坐标轴的位置
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);//求出空间点重投影到j帧图像的坐标
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();//相减得到residual //归一化相机坐标系下的重投影误差
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);//返回重投影误差均方根
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    //遍历特征库里的所有特征点
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        //特征被使用次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4) //观测少于四次点不进行外点判断
            continue;
        feature_index ++;
        //imu_i 和 imu_j是帧ID来的
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        //遍历 用到该特征点的所有帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j) //不同时刻，左相机在不同帧之间的重投影误差计算
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;   //重投影误差
                errCnt++;           //错误图像个数
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j) //不同时刻，左右图像帧之间的重投影误差
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else    //相同时刻，左右图像帧之间的重投影误差
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;  // 平均错误 = 总错误/帧数
        //FOCAL_LENGTH焦距？  错误较大时，去掉这个点
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}
//快速预测IMU？有啥用
//IMU中值积分，计算位姿与速度，注意此时的中值积分在世界坐标系下进行
// -latest_p,latest_q,latest_v,latest_acc_0,latest_gyr_0 最新时刻的姿态。
// 这个的作用是为了刷新姿态的输出，但是这个值的误差相对会比较大，是未经过非线性优化获取的初始值。
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}
//更新最新状态 这个状态latest_*是用来发topic
//获取滑窗中最新帧时刻的状态，并在世界坐标系下进行中值积分；初始化完成后，最新状态在inputIMU函数中发布
void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    // 从accBuf、gyrBuf取数据
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);    //?
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
