import random
import glob
from os import path


def take_image_files_sample(n=500, pet_family='dog'):

    path_to_original_images = path.join('data', 'images')

    # create list of image files (files with jpg extention)
    list_of_image_files = glob.glob(f'{path_to_original_images}/*.jpg')

    # files starting with lowercase are dogs
    if pet_family=='dog':
        filtered_image_files = list(filter(lambda x: x.islower(), list_of_image_files))
    elif pet_family=='cat':
        filtered_image_files = list(filter(lambda x: not x.islower(), list_of_image_files))
    else:
        raise ValueError('Wrong pet_family input. It can be either dog or cat')

    # get a sample of n image files
    return random.sample(filtered_image_files, min(n, len(filtered_image_files)))

if __name__=='__main__':
    print(take_image_files_sample(pet_family='cat'))
