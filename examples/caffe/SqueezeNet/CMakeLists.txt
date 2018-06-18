add_executable(caffe_squeezenet EXCLUDE_FROM_ALL cpp/run.cpp cpp/fp16.c cpp/fp16.h cpp/stb_image.h cpp/stb_image_resize.h)
set_source_files_properties(cpp/fp16.c PROPERTIES LANGUAGE CXX)
target_link_libraries(caffe_squeezenet mvnc)
set_target_properties(caffe_squeezenet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cpp)

add_dependencies(examples caffe_squeezenet)

download(
    https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel
    ${CMAKE_CURRENT_LIST_DIR}
    "Download caffe_squeezenet model"
    download_squeezenet_model
)

download(
    https://github.com/DeepScale/SqueezeNet/raw/master/LICENSE
    ${CMAKE_CURRENT_LIST_DIR}
    "Download caffe_squeezenet license"
    download_squeezenet_license
)
add_dependencies(download_squeezenet_model download_squeezenet_license)

add_custom_command(
    OUTPUT graph
    COMMAND mvNCCompile -w squeezenet_v1.0.caffemodel -s 12 deploy.prototxt
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS download_squeezenet_model
)
add_custom_target(compile_caffe_squeezenet DEPENDS graph)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_LIST_DIR}/graph
    ${CMAKE_CURRENT_LIST_DIR}/output.gv
    ${CMAKE_CURRENT_LIST_DIR}/output.gv.svg
    ${CMAKE_CURRENT_LIST_DIR}/output_report.html
    ${CMAKE_CURRENT_LIST_DIR}/output_expected.npy
    ${CMAKE_CURRENT_LIST_DIR}/zero_weights.caffemodel
    ${CMAKE_CURRENT_LIST_DIR}/output_result.npy
    ${CMAKE_CURRENT_LIST_DIR}/output_val.csv
)

add_dependencies(examples compile_caffe_squeezenet)
