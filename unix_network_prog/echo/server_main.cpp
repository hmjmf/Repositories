//
// Created by 启翔 张 on 2017/12/16.
//

#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <glog/logging.h>
#include <libnet.h>

void echo_back(int connect_socket){
    char recv_buf[1024];
    while (1){
        memset(&recv_buf, 0, sizeof(recv_buf));
        int rec = read(connect_socket, &recv_buf, sizeof(recv_buf));

        CHECK_NE(rec, -1) << "read";
        if(rec == 0){
            //client close
            LOG(INFO) << "close";

            break;
        }
        std::cout << recv_buf << std::endl;
        write(connect_socket, &recv_buf, strlen(recv_buf));
    }
}
int main(int argc,char* argv[]) {
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::GLOG_INFO, "./log");


    int listen_socket = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
    CHECK_GT(listen_socket, 0) << "socket";
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(6666);

    int on = 1;
    CHECK_EQ(setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)), 0) << "setscokopt";

    CHECK_EQ(bind(listen_socket,(struct sockaddr *)&server_addr, sizeof(server_addr)), 0) << "bind";

    CHECK_EQ(listen(listen_socket,SOMAXCONN), 0) << "listen";

    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);


    while(1){
        int connect_socket = accept(listen_socket,(struct sockaddr *)&client_addr, &client_addr_len);
        CHECK_GT(connect_socket, 0) << "accept";

        std::cout <<  "client->ip:" << inet_ntoa(client_addr.sin_addr) << ", port:" << ntohs(client_addr.sin_port) << std::endl;

        pid_t pid = fork();
        CHECK_NE(pid, -1) << "fork";

        if(pid == 0){
            //son
            close(listen_socket);
            echo_back(connect_socket);
            exit(0);
        } else {
            close(connect_socket);
        }

    }



    return 0;
}