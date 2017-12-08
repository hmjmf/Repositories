project(add_src)

set(HEADER_FILES
        demo_class.hpp
        )

set(SOURCE_FILES
        demo_class.cpp
        )

add_library(add_src STATIC ${SOURCE_FILES} ${HEADER_FILES})
