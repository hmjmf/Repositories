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



    int client_socket = socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
    CHECK_GT( client_socket, 0) << "socket";
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(6666);

    char send_buf[1024] = {0};
    char recv_buf[1024] = {0};

    while ( std::cin >> send_buf ){
        sendto(client_socket, send_buf, strlen(send_buf), 0, (struct sockaddr*)&server_addr, sizeof(server_addr));
        LOG(INFO) << "send:" << send_buf;
        recvfrom(client_socket, recv_buf, sizeof(recv_buf),0, NULL, NULL);
        LOG(INFO) << "recv:" << recv_buf;
        memset(recv_buf, 0 , sizeof(recv_buf));
        memset(recv_buf, 0 , sizeof(send_buf));
    }

    close(client_socket);
    return 0;
}
