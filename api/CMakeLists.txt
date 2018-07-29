file(STRINGS ${CMAKE_CURRENT_LIST_DIR}/version.txt VER_NUM)

find_package(Threads)
find_package(LibUSB REQUIRED)

file(GLOB_RECURSE SRCS "*.c")
file(GLOB_RECURSE HDRS "*.h")
file(GLOB INC_HDRS "include/*.h")
add_library(mvnc SHARED ${SRCS} ${HDRS} ${INC_HDRS})
set_target_properties(mvnc PROPERTIES LINKER_LANGUAGE C)
if(WIN32)
    target_compile_options(mvnc PRIVATE /W4)
else()
    target_compile_options(mvnc PRIVATE -O2 -Wall -MMD -MP)
endif()
target_compile_definitions(mvnc PRIVATE __PC__ USE_USB_VSC DEVICE_SHELL_ENABLED EXCLUDE_HIGHCLASS
                                        VERSION_NAME="${VER_NUM}")
target_include_directories(mvnc
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src
        ${CMAKE_CURRENT_LIST_DIR}/src/common/components/XLink/shared
        ${CMAKE_CURRENT_LIST_DIR}/src/common/components/XLink/pc
        ${CMAKE_CURRENT_LIST_DIR}/src/common/components/XLinkConsole/pc
        ${CMAKE_CURRENT_LIST_DIR}/src/common/shared/include
        ${LibUSB_INCLUDE_DIRS}
)
target_link_libraries(mvnc PRIVATE ${CMAKE_THREAD_LIBS_INIT} ${LibUSB_LIBRARIES} ${CMAKE_DL_LIBS})

file(COPY ${CMAKE_CURRENT_LIST_DIR}/src/mvnc/MvNCAPI-ma2450.mvcmd DESTINATION ${OUTPUT_DIR}/mvnc)

py_packaging(mvnc_python python ${OUTPUT_DIR})

install(TARGETS mvnc
    EXPORT mvncTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/mvnc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/mvnc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mvnc2)
install(FILES ${OUTPUT_DIR}/mvnc/MvNCAPI-ma2450.mvcmd DESTINATION ${CMAKE_INSTALL_LIBDIR}/mvnc)

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    install(FILES ${PROJECT_SOURCE_DIR}/config/97-usbboot.rules DESTINATION /etc/udev/rules.d)
    install(CODE "execute_process(COMMAND udevadm control --reload-rules COMMAND udevadm trigger)")
    configure_file(${PROJECT_SOURCE_DIR}/config/ncsdk.conf.in ${CMAKE_CURRENT_BINARY_DIR}/ncsdk.conf @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ncsdk.conf DESTINATION /etc/ld.so.conf.d)
    install(CODE "execute_process(COMMAND ldconfig)")
endif()

