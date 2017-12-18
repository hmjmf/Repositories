//
// Created by 启翔 张 on 2017/12/18.
//
#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <glog/logging.h>
#include <libnet.h>

#include "help.hpp"

int main(int argc,char* argv[]) {
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "./log_client");



    int client_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    CHECK_GT( client_socket, 0) << "socket";
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(6666);

    int ret = connect_timeout(client_socket, &server_addr, 5);
    if (ret == -1){
        if (errno == ETIMEDOUT){
            LOG(FATAL) << "time out";
        } else {
            LOG(FATAL) << "client time out";
        }
    }
    struct sockaddr_in local_addr;
    socklen_t loacl_addr_len = sizeof(local_addr);
    memset(&local_addr, 0, sizeof(local_addr));
    CHECK_GE(getsockname(client_socket, (struct sockaddr*)&local_addr, &loacl_addr_len), 0) << "getsockname";

    LOG(INFO) <<  "client->ip:" << inet_ntoa(local_addr.sin_addr) << ", port:" << ntohs(local_addr.sin_port) ;

    close(client_socket);
    return 0;
}
