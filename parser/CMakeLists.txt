PROJECT(cnn:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#ADD_EXECUTABLE(lstm-parser lstm-parse.cc)
#target_link_libraries(lstm-parser cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(lstm-parse main.cpp lstm_sdparser.cpp)
target_link_libraries(lstm-parse cnn ${Boost_LIBRARIES})
