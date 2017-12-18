//
// Created by 启翔 张 on 2017/12/18.
//

#include "help.hpp"

/*
    int ret = read_timeout(fd, seconds);
    if(ret == 0) {
        success
        read(fd, ...)
    } else if(ret == -1 && errno = ETIMEOUT) {
        TIMEOUT
    }else {
        UNKONW
    }

 */

int read_timeout(int fd, unsigned int wait_seconds){
    int ret = 0;
    if(wait_seconds > 0){
        fd_set read_fdset;
        struct timeval timeout;

        FD_ZERO(&read_fdset);
        FD_SET(fd, &read_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;
        do{
            ret = select(fd + 1, &read_fdset, NULL, NULL, &timeout);
        } while(ret < 0 && errno == EINTR);

        if(ret == 0){
            ret = -1;
            errno = ETIMEDOUT;
        } else if (ret == 1){
            ret = 0;
        }
    }
    return ret;
}
int write_timeout(int fd, unsigned int wait_seconds){
    int ret = 0;
    if(wait_seconds > 0){
        fd_set write_fdset;
        struct timeval timeout;

        FD_ZERO(&write_fdset);
        FD_SET(fd, &write_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;
        do{
            ret = select(fd + 1, NULL, &write_fdset, NULL, &timeout);
        } while(ret < 0 && errno == EINTR);

        if(ret == 0){
            ret = -1;
            errno = ETIMEDOUT;
        } else if (ret == 1){
            ret = 0;
        }
    }
    return ret;
}

int accept_timeout(int fd, struct sockaddr_in* addr, unsigned int wait_seconds){
    int ret;
    socklen_t addrlen = sizeof(struct sockaddr_in);

    if(wait_seconds > 0){
        fd_set accept_fdset;
        struct timeval timeout;

        FD_ZERO(&accept_fdset);
        FD_SET(fd, &accept_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;

        do{
            ret = select(fd + 1, &accept_fdset, NULL, NULL, &timeout);
        } while(ret < 0 && errno == EINTR);

        if(ret == -1){
            return -1;

        } else if (ret == 0){
            errno = ETIMEDOUT;
            return -1;
        }
    }

    if (addr == NULL){
        ret = accept(fd, NULL, NULL);
    } else {
        ret = accept(fd, (struct sockaddr*)addr, &addrlen);
    }

    CHECK_NE(ret, -1) << "accept";

    return ret;
}
void active_nonblock(int fd){
    int flags = fcntl(fd, F_GETFL);
    CHECK_NE(flags, -1) << "fcntl";

    flags |= O_NONBLOCK;

    CHECK_NE(fcntl(fd, F_SETFL, flags), -1) << "fcntl";
}
void deactive_nonblock(int fd){
    int flags = fcntl(fd, F_GETFL);
    CHECK_NE(flags, -1) << "fcntl";

    flags &= ~O_NONBLOCK;

    CHECK_NE(fcntl(fd, F_SETFL, flags), -1) << "fcntl";
}
int connect_timeout(int fd, struct sockaddr_in* addr, unsigned int wait_seconds){
    socklen_t addrlen = sizeof(struct sockaddr_in);

    if(wait_seconds > 0){
        //非阻塞模式
        active_nonblock(fd);
    }
    int ret = connect(fd, (struct sockaddr*)addr, addrlen);

    if (ret < 0 && errno == EINPROGRESS){
        fd_set connect_fdset;
        struct timeval timeout;

        FD_ZERO(&connect_fdset);
        FD_SET(fd, &connect_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;

        do{
            ret = select(fd + 1, NULL, &connect_fdset, NULL, &timeout);
        } while(ret < 0 && errno == EINTR);

        if (ret == 0){
            errno = ETIMEDOUT;
            ret -1;
        } else if (ret < 0){
            return -1;
        } else if (ret == 1){
            //连接成功 或 套接字出错（产生可写事件）
            int err;
            socklen_t socklen = sizeof(err);
            int sockoptret = getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &socklen);
            if (sockoptret == -1){
                return -1;
            }
            if (err == 0){
                ret = 0;
            } else {
                errno = err;
                ret = -1;
            }
        }


    }

    if(wait_seconds > 0){
        //阻塞模式
        deactive_nonblock(fd);
    }
    return ret;
}