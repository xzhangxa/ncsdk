download(
    http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/tiny-yolo-v1_53000.caffemodel
    ${CMAKE_CURRENT_LIST_DIR}
    "Download caffe_tinyyolo model"
    download_tinyyolo_model
)
download(
    http://ncs-forum-uploads.s3.amazonaws.com/ncappzoo/tiny_yolo/tiny-yolo-v1.prototxt
    ${CMAKE_CURRENT_LIST_DIR}
    "Download caffe_tinyyolo prototxt"
    download_tinyyolo_prototxt
)
add_dependencies(download_tinyyolo_model download_tinyyolo_prototxt)

add_custom_command(
    OUTPUT graph
    COMMAND mvNCCompile -w tiny-yolo-v1_53000.caffemodel -s 12 tiny-yolo-v1.prototxt
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS download_tinyyolo_model
)
add_custom_target(compile_caffe_tinyyolo DEPENDS graph)

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

add_dependencies(examples compile_caffe_tinyyolo)
