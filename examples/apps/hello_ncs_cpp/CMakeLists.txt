add_executable(hello_ncs_cpp EXCLUDE_FROM_ALL cpp/hello_ncs.cpp)
target_link_libraries(hello_ncs_cpp mvnc)
set_target_properties(hello_ncs_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cpp)

add_dependencies(examples hello_ncs_cpp)
