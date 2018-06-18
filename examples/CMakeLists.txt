add_custom_target(examples)

if(WITH_TOOLKIT)
    execute_process(COMMAND python3 -c "import caffe" RESULT_VARIABLE caffe_error ERROR_QUIET)
    if(${caffe_error} STREQUAL "0")
        add_subdirectory(caffe)
    else()
        message("")
        message("Caffe Python library is not found, Caffe models will not be downloaded/compiled when make examples.")
        message("To enable Caffe examples, install Caffe and setup PYTHONPATH before calling CMake.")
        message("")
    endif()

    add_subdirectory(tensorflow)
    add_subdirectory(data)
endif()

add_subdirectory(apps)
