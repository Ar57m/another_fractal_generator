import numpy as np
#from PIL import Image
#%pip install cython
#import cython

import cv2
from ctypes import cdll, c_double, POINTER, c_uint32, c_uint16, c_uint8



# Carregue a biblioteca
lib = cdll.LoadLibrary('./libfract.so')


fractal = lib.fractal
juliaset = lib.juliaset


lib.process_array.argtypes = [POINTER(c_uint32), POINTER(c_uint8), c_uint16, c_uint16, c_double, c_uint16, c_double]
lib.process_array.restype = None


fractal.argtypes = [POINTER(c_uint16), c_uint16, c_uint16, c_uint16, c_double, c_double, c_double, c_double]
juliaset.argtypes = [POINTER(c_uint16), c_uint16, c_uint16, c_uint16, c_double, c_double, c_double, c_double, c_double, c_double]


# Here I'm using some of my functions of my other project random_tools on github
def scale( input_tensor, new_min, new_max):
        current_min = np.min(input_tensor)
        current_max = np.max(input_tensor)
        if current_min == current_max: # to avoid infinity
           current_min -= 1
        scaled_tensor = (input_tensor - current_min) * (new_max - new_min) / (current_max - current_min) + new_min
        return scaled_tensor




# This can only scale positive numbers not negatives
def process_image(input_array, max_val, imgname):
    width, height = input_array.shape
    max = np.float64(np.max(input_array))
    max_val = np.float64(max_val)
    
    input_array = input_array.copy().reshape(-1).astype(np.uint32)
    input_array = input_array.ctypes.data_as(POINTER(c_uint32))
    output_array = (c_uint8 * (width * height* 3))()
    # Call the function
    lib.process_array(input_array, output_array, width, height, max_val, 5000, max)
    # Convert the output array to a numpy array
    output_array = np.ctypeslib.as_array(output_array).reshape(width, height, 3 )
    
    output_image = cv2.cvtColor(output_array, cv2.COLOR_BGR2RGB) 
    
    del output_array
    cv2.imwrite(f'{imgname}.png', output_image) 
    print(f'{imgname}.png' )





def image_to_array(image_path, min=0, max=2**24-1):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        array_image = np.array(img).astype(np.int64)
    
        if array_image.ndim != 3:
            array_image = np.array(img.convert("RGBA"))
            
        if array_image.ndim == 3:
              if array_image.shape[2] == 4:
                      array_image = array_image[:, :, :-1].astype(np.int64)
                      
              if array_image.shape[2] == 3:
                      array_image = (array_image[:, :, 0]*(256**2)+array_image[:, :, 1]*(256)+array_image[:, :, 2])
    
        shape = array_image.shape
        array_image = array_image.reshape(shape[0],shape[1])

        array_image = np.clip(array_image , min, max)
        
        return array_image


# Image with palette
def create_image(palette, data, filename, top_colors=4):
    data = data.copy()
    shape = data.shape
    palette = image_to_array(palette)
    unique_colors, counts = np.unique(palette, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    
    array_top_colors = unique_colors[sorted_indices][:top_colors]

    
    data = np.round(scale((np.sin(data.astype(np.float64))) , 0, array_top_colors.shape[0]-1)).reshape(-1).astype(np.uint32)
    for i, n in enumerate(array_top_colors):
        data[data == i] = n
    process_image(data.reshape(shape), np.float64(np.max(data)), filename )
 

# Not working properly
def depth_to_intensity(rgb_image, depth_map):
    depth_map = depth_map.astype(np.float64)
    normalized_depth = (depth_map / np.max(depth_map))
    
    intensity_image = rgb_image.astype(np.float64) * normalized_depth.reshape(-1) #[:,np.newaxis]
    
    return np.round(intensity_image).astype(np.uint8)





width = 4096 # I'm using ratio 1/1
height = 4096 #2304

# Here you can move around 
xmin, xmax = -16/6, 16/6   #-16/5, 16/5
ymin, ymax = -16/6, 16/6   #-9/5, 9/5

max_iter = 1000

# How many top colors to use from the palette.png
top_colors = 6

juliaset_c_real = -0.8
juliaset_c_imag = 0.16




import time

# Mandelbrot Set
start_time = time.perf_counter()

mandelbrot_set = np.empty((height, width), dtype=np.uint16)
fractal(mandelbrot_set.ctypes.data_as(POINTER(c_uint16)), width, height, max_iter, xmin, xmax, ymin, ymax)

end_time = time.perf_counter()

print("Took ", end_time - start_time, "seconds to generate")
process_image(mandelbrot_set, (2**24-1), "generated_fractal" )
create_image("palette.png",mandelbrot_set.reshape(width, height), "colorful", top_colors=top_colors)


# Julia Set
start_time = time.perf_counter()

julia_set = np.empty((height, width), dtype=np.uint16)
juliaset(julia_set.ctypes.data_as(POINTER(c_uint16)), width, height, max_iter, juliaset_c_real, juliaset_c_imag, xmin, xmax, ymin, ymax)

end_time = time.perf_counter()

print("Took ", end_time - start_time, "seconds to generate")
process_image(julia_set, (2**24-1), "generated_fractal_julia_set" )
create_image("palette.png",julia_set.reshape(width, height), "colorful_julia_set", top_colors=top_colors)




