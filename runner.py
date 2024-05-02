import numpy as np
from PIL import Image
#%pip install cython
#import cython
#import pybind11

from ctypes import cdll, c_int, c_double, POINTER



# Carregue a biblioteca
lib = cdll.LoadLibrary('./libfract.so')


fractal = lib.fractal
juliaset = lib.juliaset
fractal.argtypes = [POINTER(c_int), c_int, c_int, c_int, c_double, c_double, c_double, c_double]
juliaset.argtypes = [POINTER(c_int), c_int, c_int, c_int, c_double, c_double, c_double, c_double, c_double, c_double]


# Here I'm using some of my functions of my other project random_tools on github
def scale( input_tensor, new_min, new_max):
        current_min = np.min(input_tensor)
        current_max = np.max(input_tensor)
        if current_min == current_max: # to avoid infinity
           current_min -= 1
        scaled_tensor = (input_tensor - current_min) * (new_max - new_min) / (current_max - current_min) + new_min
        return scaled_tensor


def tensor_to_image(tensor, imgname, newmin= None, newmax= None):
        shape = tensor.shape
        array_image = np.nan_to_num(tensor.copy(), nan=0, posinf=1, neginf=-1)
        del tensor
        if (newmin == None) or (newmax == None):
            array_image = scale(array_image, 0, 16777215)
        else:
            array_image = scale(array_image, newmin, newmax)
        array_image = np.round(array_image).astype(np.int64)
        hexa =None

        hexa = array_image
        del array_image
    
        hexa_flat = hexa.reshape(-1)
        del hexa
        cor_rgb_flat = np.zeros((hexa_flat.size, 3), dtype=np.int64)
        cor_rgb_flat[:, 0] = (hexa_flat >> 16) & 0xFF
        cor_rgb_flat[:, 1] = (hexa_flat >> 8) & 0xFF
        cor_rgb_flat[:, 2] = hexa_flat & 0xFF
      
        del hexa_flat
    
        cor_rgb = cor_rgb_flat.reshape(shape[0], shape[1], 3).astype(np.uint8)
        del cor_rgb_flat 
        #oo = np.full((sh[0], sh[1], 1), 255, dtype=np.uint8)
        #cor_rgb = np.concatenate((cor_rgb, oo), axis=2)
    
        imag = Image.fromarray(cor_rgb)
        del cor_rgb
        imag.save(f'{imgname}.png')
        print(f'{imgname}.png' )


def image_to_array(image_path, min=0, max=2**24-1):
        img = Image.open(image_path)
    
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



def create_image(palette, data, filename, top_colors=4):
    data = data.copy() 
    shape = data.shape
    palette = image_to_array(palette)
    unique_colors, counts = np.unique(palette, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    
    array_top_colors = unique_colors[sorted_indices][:top_colors]


    data = np.round(scale((np.sin(data)) , 0, array_top_colors.shape[0]-1)).reshape(-1)
    for i, n in enumerate(array_top_colors):
        data[data == i] = n

    tensor_to_image(data.reshape(shape), filename, np.min(data),np.max(data))




width = 4096 # I'm using ratio 16/9
height = 4096 #2304

# Here you can move around 
xmin, xmax = -16/5, 16/5
ymin, ymax = -16/5, 16/5

max_iter = 1000

import time
start_time = time.perf_counter()

mandelbrot_set = np.empty((height, width), dtype=np.int32)
fractal(mandelbrot_set.ctypes.data_as(POINTER(c_int)), width, height, max_iter, xmin, xmax, ymin, ymax)

end_time = time.perf_counter()

print("Took ", end_time - start_time, "seconds to generate")


create_image("palette.png",mandelbrot_set, "colorful", top_colors=7)
tensor_to_image(mandelbrot_set,"generated_fractal", 0,2**24-1)




start_time = time.perf_counter()

julia_set = np.empty((height, width), dtype=np.int32)
juliaset(julia_set.ctypes.data_as(POINTER(c_int)), width, height, max_iter, -0.257443, 0.659694, xmin, xmax, ymin, ymax)

end_time = time.perf_counter()

print("Took ", end_time - start_time, "seconds to generate")


create_image("palette.png",julia_set, "colorful_julia_set", top_colors=7)
tensor_to_image(julia_set, "generated_fractal_julia_set", 0,2**24-1)






