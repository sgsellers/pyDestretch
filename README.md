# pyDestretch
Destretch algorithm for solar images

## Usage

Destretch algorithm for use with solar image data to either remove first-order atmospheric distortions, or correct reconstructed images for atmospherically-induced motions. 

Initializing the "Destretch" class by providing a reference image and series of kernel sizes will calculate and perform an affine-based warp on the target image with the "Destretch.perform_destretch()" function.

Initializing the "Destretch" class with an array of warp vectors (such as those calculated by the Destretch class) will warp a target image by those vectors when the Destretch.perform_destretch() function is called. 

Passing the "ncores" kwarg to the Destretch class will allow the routine to be parallelized using a multiprocessing Pool. Passing "return_vectors=True" will return the warp vectors after "perform_destretch()". This allows the user to, for example, calculate destretch vectors on a continuum camera channel, then dewarp a cotemporal chromospheric image using the same vectors, allowing the user to preserve chromospheric motions.

The included folder of test_destretch_data contains post-speckle reconstructed data from the Dunn Solar Telescope (DST), for use with the destretch_examples.py example script, showing how the class is initialized and used.
