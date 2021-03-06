//
// Created by 启翔 张 on 2017/12/16.
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

    CHECK_EQ(connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)),0) << "connect";

    struct packet send_packet;
    struct packet recv_packet;
    memset(&send_packet, 0, sizeof(send_packet));
    memset(&recv_packet, 0, sizeof(recv_packet));

    fd_set rset;
    FD_ZERO(&rset);

    int maxfd;
    int fd_stdin = fileno(stdin);
    maxfd = (fd_stdin > client_socket ? fd_stdin : client_socket) + 1;

    while(1){
        FD_SET(fd_stdin, &rset);
        FD_SET(client_socket, &rset);
        int nready = select(maxfd, &rset, NULL, NULL, NULL);

        CHECK_NE(nready, -1) << "select";

        if(nready == 0){
            continue;
        } else if(FD_ISSET(client_socket, &rset)){
            //接收
            memset(&recv_packet, 0, sizeof(recv_packet));
            int rec = readn(client_socket, &recv_packet.len, sizeof(recv_packet.len));
            CHECK_NE(rec, -1) << "read head fail";
            if(rec < sizeof(recv_packet.len)){
                LOG(INFO) << "server close when read head";
                break;
            } else {
                //读数据
                int n = ntohl(recv_packet.len);
                rec = readn(client_socket, &recv_packet.buf, n);
                CHECK_NE(rec, -1) << "read data fail";
                if(rec < n){
                    LOG(INFO) << "server close when read data";
                    break;
                } else {
                    LOG(INFO) << "receive:" << recv_packet.buf;
                }
            }
        } else if(FD_ISSET(fd_stdin, &rset)){
            //发送
            memset(&send_packet, 0, sizeof(send_packet));
            if((std::cin >> send_packet.buf)){
                int n = strlen(send_packet.buf);
                send_packet.len = htonl(n);
                writen(client_socket, &send_packet, sizeof(send_packet.len) + n);
            } else {
                LOG(INFO) << "shutdown";
                shutdown(client_socket, SHUT_WR);
            }
        } else {
            //error
            close(client_socket);
            LOG(FATAL) << "UN KNOW fileno";
            break;
        }
    }

    close(client_socket);
    return 0;
}