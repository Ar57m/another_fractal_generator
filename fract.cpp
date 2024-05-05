// fract.cpp
#include <omp.h>
#include <math.h>
#include <cstdint>

#include <cmath>
#include <vector>


extern "C" {

    void fractal(uint16_t* output, uint16_t width, uint16_t height, uint16_t max_iter, double xmin=-2, double xmax=2, double ymin=-2, double ymax=2) {
        double dx = (xmax - xmin) / width, dy = (ymax - ymin) / height;

        #pragma omp parallel for schedule(dynamic)
        for (uint16_t x = 0; x < width; ++x) {

            double r_part = 0;
            
            r_part = xmin + x * dx;

            for (uint16_t y = 0; y < height; ++y) {
                
                double i_part = ymin + y * dy;
                double z_real = 0;
                double z_imag = 0;

                double c_real = 0;
                double c_imag = 0;
                c_real = r_part;
                c_imag = i_part;
                uint16_t iteration = 0;


                while (z_real * z_real + z_imag * z_imag < 4 && iteration < max_iter) {
                    double temp = z_real * z_real - z_imag * z_imag + c_real;
                    z_imag = 2 * z_real * z_imag + c_imag;
                    z_real = 0;
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
            double r_part = 0; 
            r_part = xmin + x * dx;

            for (uint16_t y = 0; y < height; ++y) {
                double i_part = ymin + y * dy;

                double z_real = 0;
                double z_imag = 0;
                z_real = r_part;
                z_imag = i_part;
                uint16_t iteration = 0;


                while (z_real * z_real + z_imag * z_imag < 4 && iteration < max_iter) {
                    double temp = z_real * z_real - z_imag * z_imag + c_real;
                    z_imag = 2 * z_real * z_imag + c_imag;
                    z_real = 0;
                    z_real = temp;
                    ++iteration;
                }
                output[y * width + x] = iteration;
            }
        }
    }

}




extern "C" {
    void process_array(uint32_t* input_array, uint8_t* output_array, uint16_t width, uint16_t height, double max_value, uint16_t batch_size, double npmax) {
        // Iterate over each batch
        for(int i = 0; i < width * height; i += batch_size) {
            // Iterate over each value in the batch
            for(int j = i; j < i + batch_size && j < width * height; j++) {
                // Convert the value to double and scale it
                double value = (static_cast<double>((input_array[j])) / npmax) * max_value;

                // Round to nearest integer
                uint32_t rounded_value = static_cast<uint32_t>(std::round(value));
                
                // Separate RGB channels
                output_array[j * 3] = (rounded_value >> 16) & 0xFF;
                output_array[j * 3 + 1] = (rounded_value >> 8) & 0xFF;
                output_array[j * 3 + 2] = rounded_value & 0xFF;
            }
        }
    }
}