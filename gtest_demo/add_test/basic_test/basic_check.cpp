//
// Created by 启翔 张 on 2017/11/26.
//

//#include "../lib/googletest/include/gtest/gtest.h"
#include "gtest/gtest.h"
#include "demo_class.hpp"
//命名最好不要在头尾有下划线
TEST(basic_check, test_eq){
    EXPECT_EQ(1, 1);
}

TEST(basic_check, test_neq){
    EXPECT_NE(1, 0);
}
TEST(basic_check, test_add){
    EXPECT_EQ(1, add2(0,1));
}


//-------class test

class classTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        a = 15;
    }
    virtual void TearDown(){

    }
    int a;
};

TEST_F(classTest, class_test){
    EXPECT_EQ(15,a);
}

//-------参数化
class IsPrimeParamTest : public::testing::TestWithParam<int>
{
};
TEST_P(IsPrimeParamTest, Negative){
    int n =  GetParam();
    EXPECT_FALSE(IsPrime(n));
}

INSTANTIATE_TEST_CASE_P(NegativeTest, IsPrimeParamTest, testing::Values(-1,-2,-5,-100,INT_MIN));

//-------类型参数化
template <class T>
class vectorTest:public testing::Test{
protected:
    vectorTest(){
        v = CreateVector<T>();

    }
    std::vector<T>* v;
};
using testing::Types;
typedef Types<int, char> Implementations;
TYPED_TEST_CASE(vectorTest, Implementations);
TYPED_TEST(vectorTest, DefaultConstructor) {
    EXPECT_EQ(0u, this->v->size());
}

//-or
template <class T>
class vectorTest2:public testing::Test{
protected:
    vectorTest2(){
        v = CreateVector<T>();

    }
    std::vector<T>* v;
};
TYPED_TEST_CASE_P(vectorTest2);
TYPED_TEST_P(vectorTest2, DefaultConstructor2) {
    EXPECT_EQ(0u, this->v->size());
}

REGISTER_TYPED_TEST_CASE_P(vectorTest2, DefaultConstructor2);
typedef Types<int, char> Implementations;

INSTANTIATE_TYPED_TEST_CASE_P(QueueInt_Char, vectorTest2, Implementations);

//-------共享变量
class FooEnvironment: public testing::Environment
{
public:
    virtual void SetUp()
    {
        printf("Environment SetUp!\n");
        a = 100;
    }
    virtual void TearDown()
    {
        printf("Environment TearDown!\n");
    }
    int a;     //共享数据
};
FooEnvironment* foo_env;  //对象指针声明
TEST(firstTest, first){
    EXPECT_EQ(100,foo_env->a);
    foo_env->a ++;
}
TEST(secondTest, second){
    EXPECT_EQ(101,foo_env->a);
}



int main(int argc, char* argv[])
{
    foo_env = new FooEnvironment;
    testing::AddGlobalTestEnvironment(foo_env);     //注册
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}