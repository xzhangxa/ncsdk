py_packaging(mvnctools_lib mvnctools-lib ${OUTPUT_DIR})
py_packaging(mvnctools_check mvnctools-check ${OUTPUT_DIR})
py_packaging(mvnctools_compile mvnctools-compile ${OUTPUT_DIR})
py_packaging(mvnctools_profile mvnctools-profile ${OUTPUT_DIR})

add_dependencies(mvnctools_lib mvnc_python)
add_dependencies(mvnctools_check mvnctools_lib)
add_dependencies(mvnctools_compile mvnctools_lib)
add_dependencies(mvnctools_profile mvnctools_lib)
