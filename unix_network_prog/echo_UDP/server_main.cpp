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
#include <signal.h>

void echo_back(int socket){
    char recvbuf[1024] = {0};
    struct sockaddr_in peeraddr;
    socklen_t peerlen = sizeof(sockaddr_in);
    while (1){
        memset(recvbuf, 0, sizeof(recvbuf));
        int n = recvfrom(socket, recvbuf, sizeof(recvbuf), 0, (struct sockaddr*)&peeraddr, &peerlen);

        if (n == -1){
            CHECK_EQ(errno, EINTR) << "recvfrom";
            continue;
        } else if (n > 0){
            LOG(INFO) << "send:" << recvbuf;
            sendto(socket, recvbuf, n, 0, (struct sockaddr*)&peeraddr, peerlen);
        }
    }
    close(socket);

}

int main(int argc,char* argv[]) {
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "./log_server");



    int listen_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    CHECK_GT(listen_socket, 0) << "socket";
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(6666);

    int on = 1;
    CHECK_EQ(setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)), 0) << "setscokopt";

    CHECK_EQ(bind(listen_socket,(struct sockaddr *)&server_addr, sizeof(server_addr)), 0) << "bind";

    echo_back(listen_socket);

    return 0;
}