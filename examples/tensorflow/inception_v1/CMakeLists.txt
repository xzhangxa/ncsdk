download_extract(
    http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
    ${CMAKE_CURRENT_LIST_DIR}
    "Download inception_v1"
    download_inception_v1
)
add_custom_command(
    TARGET download_inception_v1
    POST_BUILD
    COMMAND python3 inception-v1.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

add_custom_command(
    OUTPUT graph
    COMMAND mvNCCompile -s 12 output/inception-v1.meta -in=input -on=InceptionV1/Logits/Predictions/Reshape_1
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS download_inception_v1
)
add_custom_target(compile_inception_v1 DEPENDS graph)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_LIST_DIR}/graph
    ${CMAKE_CURRENT_LIST_DIR}/output.gv
    ${CMAKE_CURRENT_LIST_DIR}/output.gv.svg
    ${CMAKE_CURRENT_LIST_DIR}/output_report.html
    ${CMAKE_CURRENT_LIST_DIR}/output_expected.npy
    ${CMAKE_CURRENT_LIST_DIR}/inception_v1.ckpt
    ${CMAKE_CURRENT_LIST_DIR}/output_result.npy
    ${CMAKE_CURRENT_LIST_DIR}/output_val.csv
    ${CMAKE_CURRENT_LIST_DIR}/output
)

add_dependencies(examples compile_inception_v1)

