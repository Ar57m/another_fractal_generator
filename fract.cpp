// fract.cpp
#include <omp.h>
#include <math.h>

extern "C" {

    void fractal(int* output, int width, int height, int max_iter, double xmin=-2, double xmax=2, double ymin=-2, double ymax=2) {
        double dx = (xmax - xmin) / width;
        double dy = (ymax - ymin) / height;
        
        #pragma omp parallel for
        for (int x = 0; x < width; ++x) {
            double r_part = xmin + x * dx;
            for (int y = 0; y < height; ++y) {
                double i_part = ymin + y * dy;
                double z_real = 0, z_imag = 0, c_real = r_part, c_imag = i_part;
                int iteration = 0;
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

    void juliaset(int* output, int width, int height, int max_iter, double c_real, double c_imag, double xmin=-2, double xmax=2, double ymin=-2, double ymax=2) {
        double dx = (xmax - xmin) / width;
        double dy = (ymax - ymin) / height;
        
        #pragma omp parallel for
        for (int x = 0; x < width; ++x) {
            double r_part = xmin + x * dx;
            for (int y = 0; y < height; ++y) {
                double i_part = ymin + y * dy;
                double z_real = r_part, z_imag = i_part;
                int iteration = 0;
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