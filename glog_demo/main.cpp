#include <iostream>
#include <string>
#include <fstream>
#include <glog/logging.h>

void FatalProcessFunction() {
    std::cout<<"FATAL PROCESS HERE"<<std::endl;
    exit(1);
}

int main(int argc,char* argv[]) {
    google::InitGoogleLogging(argv[0]); //初始化 glog
    google::SetLogDestination(google::GLOG_FATAL, "./log"); //level 3
    google::SetLogDestination(google::GLOG_ERROR, "./log"); //level 2
    google::SetLogDestination(google::GLOG_WARNING, "./log"); //level 1
    google::SetLogDestination(google::GLOG_INFO, "./log"); //level 0

    LOG(INFO) << "INFO LOG";
    LOG(WARNING) << "WARNING LOG";
    LOG(ERROR) << "ERROR LOG";
    //LOG(FATAL) << "FATAL LOG";//  will exit


    VLOG(100) << "VLOG INFO 100";//100 is log level ,-v 100+

    LOG_IF(INFO, 11 > 10) << "11 > 10";
    LOG_IF(INFO, 11 < 10) << "11 < 10";

    for(int i=0;i<10;i++){
        //3次log一次
        LOG_EVERY_N(INFO, 3) << "Got the " << google::COUNTER << "th";
    }

    for(int i=0;i<10;i++){
        //前5次log
        LOG_FIRST_N(INFO, 5) << "Got the " << google::COUNTER << "th";
    }

    DLOG(INFO) << "DEBUG";//log if debug

    CHECK( 1 > 0 ) << "1 shuld be < 0"; // if wrong then exit

    CHECK_EQ(1,1)  << " "; // 1==1
    CHECK_DOUBLE_EQ(1.0,1.0);
    CHECK_NE(1,2) << " "; // 1!=2


    CHECK_GE(2,1)  << " "; //2>=1
    CHECK_GT(2,1)  << " "; //2>1

    CHECK_LE(1,2)  << " "; //1<=2
    CHECK_LT(1,2)  << " "; //1<2

    int a = 10;
    int* pa = &a;
    CHECK_NOTNULL(pa);

    int arr[10];
    CHECK_INDEX(9,arr);//9 < (sizeof(arr)/sizeof(arr[0]))

    CHECK_BOUND(10,arr); //10 <= (sizeof(arr)/sizeof(arr[0]))

    CHECK_NEAR(1,2,3); // 1 >= 2-3 && 1<=2+3

    CHECK_ERR(1); // CHECK >=0

    google::InstallFailureFunction(&FatalProcessFunction);
    LOG(FATAL) << "FATAL LOG "; //  will run  FatalProcessFunction & log & EXIT



//    CHECK_OP(name, op, val1, val2);
//    CHECK_STRCASENE(s1, s2);
//    CHECK_STRCASENE(s1, s2);
//    CHECK_STREQ(s1, s2);
//    CHECK_STRNE(s1, s2);


    return 0;
}