include_directories (#${SOURCE_DIR}/
    # ${THIRDPARTY_DIR}/boost/include
     ${PROJECT_SOURCE_DIR}/eigen-3.2.4/
    #${THIRDPARTY_DIR}/cnn/
    )

set (lstm_sdparser_VERSION "0.0.1")
set (lstm_sdparser_SRC
    lstm_sdparser.cpp
    )

# look for Boost
if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


add_library(lstm_sdparser_static_lib STATIC lstm_sdparser_dll.cpp ${lstm_sdparser_SRC})
set_target_properties (lstm_sdparser_static_lib    PROPERTIES
    OUTPUT_NAME     lstm_sdparser)

add_library(lstm_sdparser_shared_lib SHARED lstm_sdparser_dll.cpp ${lstm_sdparser_SRC})
set_target_properties (lstm_sdparser_shared_lib    PROPERTIES
    VERSION         ${lstm_sdparser_VERSION}
    OUTPUT_NAME     lstm_sdparser)

# redirect output binary to tools/train
add_executable(lstm-parse main.cpp ${lstm_sdparser_SRC})
target_link_libraries(lstm-parse cnn ${Boost_LIBRARIES})
set_target_properties (lstm-parse PROPERTIES
    OUTPUT_NAME lstm-parse
    RUNTIME_OUTPUT_DIRECTORY train/)
configure_file (
    lstm_sdparser_dll.h
    ltp/lstm_sdparser_dll.h)


