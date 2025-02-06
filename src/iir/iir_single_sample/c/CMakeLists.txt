cmake_minimum_required(VERSION 3.29)
project(iir_single_sample C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(iir_single_sample
        iir_single_sample.c
        iir_single_sample.h
        main.c)

# Add custom target for running
add_custom_target(run
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/iir_single_sample
        DEPENDS iir_single_sample
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running executable..."
)