cmake_minimum_required(VERSION 2.8)
project(ncsdk)

set(ncsdk_VERSION_MAJOR 1)
set(ncsdk_VERSION_MINOR 12)
set(ncsdk_VERSION_PATCH 0)

if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif()

# set CMAKE_INSTALL_* if not defined
include(GNUInstallDirs)

set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/output CACHE PATH "Output Directory")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

# force default mode as Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

option(WITH_TOOLKIT "Include toolkit" ON)

include(${PROJECT_SOURCE_DIR}/cmake/download.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/python.cmake)

add_subdirectory(api)

if(WITH_TOOLKIT)
    add_subdirectory(tk)
endif()

add_subdirectory(examples)

include(ExternalProject)
ExternalProject_Add(
    caffe
    GIT_REPOSITORY https://github.com/weiliu89/caffe.git
    GIT_TAG ssd
    CMAKE_ARGS -Dpython_version=3 -DCPU_ONLY=ON -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/caffe
    PREFIX ${PROJECT_SOURCE_DIR}/caffe
    EXCLUDE_FROM_ALL TRUE
)
add_custom_target(pycaffe)
add_dependencies(pycaffe caffe)
add_custom_command(
    TARGET pycaffe
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "\n"
    COMMAND ${CMAKE_COMMAND} -E echo "Set PYTHONPATH to ${PROJECT_SOURCE_DIR}/caffe/python to use Caffe Python library"
    COMMAND ${CMAKE_COMMAND} -E echo "\n"
)

# setup/install project CMake config file
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/ncsdk)
include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/templates/ncsdkConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/ncsdkConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/ncsdkConfigVersion.cmake"
    VERSION ${ncsdk_VERSION_MAJOR}.${ncsdk_VERSION_MINOR}.${ncsdk_VERSION_PATCH} COMPATIBILITY AnyNewerVersion)
export(EXPORT mvncTargets NAMESPACE ncsdk:: FILE "${PROJECT_BINARY_DIR}/ncsdkTargets.cmake")
install(EXPORT mvncTargets NAMESPACE ncsdk:: FILE ncsdkTargets.cmake DESTINATION ${INSTALL_CONFIG_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ncsdkConfig.cmake DESTINATION ${INSTALL_CONFIG_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ncsdkConfigVersion.cmake DESTINATION ${INSTALL_CONFIG_DIR})

# uninstall target
# from CMake official wiki https://gitlab.kitware.com/cmake/community/wikis/FAQ#can-i-do-make-uninstall-with-cmake
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
