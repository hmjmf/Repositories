//
// Created by 启翔 张 on 2017/11/26.
//

#ifndef GOOGLE_TEST_DEMO_DEMO_CLASS_HPP
#define GOOGLE_TEST_DEMO_DEMO_CLASS_HPP

#include <vector>
class demo_class {
public:
    static int add(int a,int b);
};

int add2(int a,int b){
    return a + b;
}
bool IsPrime(int n){
    return n > 0;
}

template <class T>
std::vector<T>* CreateVector();

template <>
std::vector<int>* CreateVector(){
    return new std::vector<int>;
}

template <>
std::vector<char>* CreateVector(){
    return new std::vector<char>;
}

#endif //GOOGLE_TEST_DEMO_DEMO_CLASS_HPP
