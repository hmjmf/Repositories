//
// Created by 启翔 张 on 2017/12/18.
//

#ifndef CLIENT_HELP_HPP
#define CLIENT_HELP_HPP

#include <iostream>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <glog/logging.h>
#include <libnet.h>

void active_nonblock(int fd);
void deactive_nonblock(int fd);

int read_timeout(int fd, unsigned int wait_seconds);
int write_timeout(int fd, unsigned int wait_seconds);
int accept_timeout(int fd, struct sockaddr_in* addr, unsigned int wait_seconds);
int connect_timeout(int fd, struct sockaddr_in* addr, unsigned int wait_seconds);

#endif //CLIENT_HELP_HPP
