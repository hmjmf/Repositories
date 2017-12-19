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

ssize_t	 readn(int fildes, void *buf, size_t nbyte){
    size_t nleft = nbyte;
    ssize_t nread;
    char* bufp = (char*)buf;
    while (nleft > 0){
        nread = read(fildes, bufp, nbyte);
        if(nread < 0){
            if (errno == EINTR){
                //中断,未传完
                continue;
            } else {
                return -1;
            }
        } else if (nread == 0){
            //关闭
            return nbyte - nleft;
        } else {
            bufp += nread;
            nleft -= nread;
        }
    }
    return nbyte;
}
ssize_t	 writen(int fildes, const void *buf, size_t nbyte){
    size_t nleft = nbyte;
    ssize_t nwrite;
    char* bufp = (char*)buf;
    while (nleft > 0){
        nwrite = write(fildes, bufp, nbyte);
        if(nwrite < 0){
            if (errno == EINTR){
                //中断,未传完
                continue;
            } else {
                return -1;
            }
        } else if (nwrite == 0){
            continue;
        } else {
            bufp += nwrite;
            nleft -= nwrite;
        }
    }
    return nbyte;
}