add_custom_command(
    OUTPUT graph
    COMMAND mvNCCompile -w MobileNetSSD_deploy.caffemodel -s 12 patched_MobileNetSSD_deploy.prototxt
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS download_mobilenet_ssd_model
)
add_custom_target(compile_caffe_mobilenet_ssd DEPENDS graph)

set(PYTHON_EXECUTABLE python3)
if(WIN32)
    set(PYTHON_EXECUTABLE python)
endif()
add_custom_command(
    OUTPUT MobileNetSSD_deploy.caffemodel
    COMMAND ${PYTHON_EXECUTABLE} download_mobilenet_ssd.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
add_custom_target(download_mobilenet_ssd_model DEPENDS MobileNetSSD_deploy.caffemodel)

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

#add_dependencies(examples compile_caffe_mobilenet_ssd)
