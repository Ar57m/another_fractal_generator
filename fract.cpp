// fract.cpp
#include <omp.h>
#include <math.h>
#include <cstdint>
#include <limits>
#include <iostream>

extern "C" {

    void fractal(uint16_t* output, uint16_t width, uint16_t height, uint16_t max_iter, double xmin=-2, double xmax=2, double ymin=-2, double ymax=2) {
        double dx = (xmax - xmin) / width, dy = (ymax - ymin) / height;

        #pragma omp parallel for schedule(dynamic)
        for (uint16_t x = 0; x < width; ++x) {
            double r_part = xmin + x * dx;
            for (uint16_t y = 0; y < height; ++y) {
                double i_part = ymin + y * dy;
                double z_real = 0;
                double z_imag = 0;
                double c_real = r_part;
                double c_imag = i_part;
                uint16_t iteration = 0;
                while (z_real * z_real + z_imag * z_imag < 4 && iteration < max_iter) {
                    double temp = z_real * z_real - z_imag * z_imag + c_real;
                    z_imag = 2 * z_real * z_imag + c_imag;
                    z_real = temp;
                    ++iteration;
                }
                output[y * width + x] = iteration;
            }
        }
    }

}

extern "C" {

    void juliaset(uint16_t* output, uint16_t width, uint16_t height, uint16_t max_iter, double c_real, double c_imag, double xmin=-2, double xmax=2, double ymin=-2, double ymax=2) {
        double dx = (xmax - xmin) / width, dy = (ymax - ymin) / height;
        
        #pragma omp parallel for schedule(dynamic)
        for (uint16_t x = 0; x < width; ++x) {
            double r_part = xmin + x * dx;
            for (uint16_t y = 0; y < height; ++y) {
                double i_part = ymin + y * dy;
                double z_real = r_part;
                double z_imag = i_part;
                uint16_t iteration = 0;
                while (z_real * z_real + z_imag * z_imag < 4 && iteration < max_iter) {
                    double temp = z_real * z_real - z_imag * z_imag + c_real;
                    z_imag = 2 * z_real * z_imag + c_imag;
                    z_real = temp;
                    ++iteration;
                }
                output[y * width + x] = iteration;
            }
        }
    }

}