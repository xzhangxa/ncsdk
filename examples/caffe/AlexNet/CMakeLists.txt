add_executable(caffe_alexnet EXCLUDE_FROM_ALL cpp/run.cpp cpp/stb_image.h cpp/stb_image_resize.h)
target_link_libraries(caffe_alexnet mvnc)
set_target_properties(caffe_alexnet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cpp)

add_dependencies(examples caffe_alexnet)

download(
    http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
    ${CMAKE_CURRENT_LIST_DIR}
    "Download caffe_alexnet model"
    download_alexnet_model
)

add_custom_command(
    OUTPUT graph
    COMMAND mvNCCompile -w bvlc_alexnet.caffemodel -s 12 deploy.prototxt
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS download_alexnet_model
)
add_custom_target(compile_caffe_alexnet DEPENDS graph)

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

add_dependencies(examples compile_caffe_alexnet)
