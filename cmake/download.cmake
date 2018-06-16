set(realpath ${CMAKE_CURRENT_LIST_DIR})

function(download url dest msg target depend_by)
    get_filename_component(filename ${url} NAME)
    get_filename_component(dest_abs ${dest} ABSOLUTE)
    set(URL ${url})
    set(DEST ${dest_abs}/${filename})
    set(MSG ${msg})
    configure_file(${realpath}/templates/download.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${target}/download.cmake @ONLY)
    add_custom_target(${target} COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${target}/download.cmake)
    foreach(d ${depend_by})
        add_dependencies(${d} ${target})
    endforeach()
endfunction()

function(download_extract url dest msg target upper_target)
    get_filename_component(filename ${url} NAME)
    get_filename_component(dest_abs ${dest} ABSOLUTE)
    download(${url} ${dest} ${msg} ${target} ${upper_target})
    add_custom_command(
        TARGET ${target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E tar xf ${dest_abs}/${filename}
        WORKING_DIRECTORY ${dest}
    )
endfunction()
