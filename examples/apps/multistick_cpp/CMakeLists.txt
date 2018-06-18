add_executable(multistick_cpp EXCLUDE_FROM_ALL cpp/multistick.cpp cpp/fp16.c cpp/fp16.h cpp/stb_image.h cpp/stb_image_resize.h)
set_source_files_properties(cpp/fp16.c PROPERTIES LANGUAGE CXX)
target_link_libraries(multistick_cpp mvnc)
set_target_properties(multistick_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cpp)

add_dependencies(examples multistick_cpp)

add_custom_target(setup_multistick_cpp
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/../../caffe/GoogLeNet/graph ${CMAKE_CURRENT_LIST_DIR}/cpp/googlenet.graph
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/../../caffe/SqueezeNet/graph ${CMAKE_CURRENT_LIST_DIR}/cpp/squeezenet.graph
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cpp
)
add_dependencies(setup_multistick_cpp compile_caffe_googlenet)
add_dependencies(setup_multistick_cpp compile_caffe_squeezenet)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_LIST_DIR}/cpp/googlenet.graph
    ${CMAKE_CURRENT_LIST_DIR}/cpp/squeezenet.graph
)

add_dependencies(examples setup_multistick_cpp)
