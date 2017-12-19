//
// Created by 启翔 张 on 2017/12/16.
//

#ifndef CLIENT_HELP_HPP
#define CLIENT_HELP_HPP

#endif //CLIENT_HELP_HPP

ssize_t	 writen(int fildes, const void *buf, size_t nbyte);
ssize_t	 readn(int fildes, void *buf, size_t nbyte);

struct packet{
    int len;
    char buf[1024];
};