cmake_minimum_required(VERSION 3.29)
project(adaptive_single_sample C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(adaptive_single_sample
        adaptive_single_sample.c
        adaptive_single_sample.h
        main.c)
