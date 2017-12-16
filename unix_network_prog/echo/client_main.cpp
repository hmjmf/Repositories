//
// Created by 启翔 张 on 2017/12/16.
//

#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <glog/logging.h>
#include <libnet.h>

int main() {
    int client_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    CHECK_GT( client_socket, 0) << "socket";
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(6666);

    CHECK_EQ(connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)),0) << "connect";


    char send_buf[1024] = {0};
    char recv_buf[1024] = {0};
    while(std::cin >> send_buf){
        write(client_socket, send_buf, strlen(send_buf));

        memset(recv_buf, 0, sizeof(recv_buf));
        read(client_socket, recv_buf, sizeof(recv_buf));
        std::cout << recv_buf << std::endl;
    }





    return 0;
}