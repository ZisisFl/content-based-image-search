import cv2
import numpy as np
import pickle
from tqdm import tqdm
from os import path
from matplotlib import pyplot as plt
from images_sampling import take_image_files_sample


class PetImage:
    
    def __init__(self, filename) -> None:
        self.image_in_opencv_format = cv2.imread(filename)
        self.feature_vector = self.extract_feature_vector(self.image_in_opencv_format)
        self.family = self.extract_pet_family(filename)
        self.breed = self.extract_pet_breed(filename)
        self.filename = path.split(filename)[1]
        self.path_to_images_folder = path.split(filename)[0]
        

    def encode_feature_vector(self):
        self.feature_vector = pickle.dumps(self.feature_vector)
    

    def decode_feature_vector(self):
        self.feature_vector = pickle.loads(self.feature_vector)
    

    def extract_pet_family(self, image_filename):
        if image_filename.islower():
            pet_family = 'dog'
        else:
            pet_family = 'cat'

        return pet_family
    
    
    def extract_pet_breed(self, image_filename):
        return '_'.join(image_filename.lower().split('/')[-1].split('_')[:-1])
    

    def extract_feature_vector(self, image, vector_size=16):
        try:
            alg = cv2.SIFT_create()
            # Dinding image keypoints
            kps = alg.detect(image)

            # Getting first vector_size of them. 
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            
            # computing descriptors vector
            kps, dsc = alg.compute(image, kps)

            # Flatten all of them in one big vector - our feature vector
            dsc = dsc.flatten()
            # Making descriptor of same size
            # Descriptor vector size is 128
            needed_size = (vector_size * 128)
            if dsc.size < needed_size:
                # if we have less the vector_size descriptors then just adding zeros at the
                # end of our feature vector
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except Exception as e:
            print(f'Could not extract features from image due to error: {e}')
            return None

        return dsc
    

    def to_dict(self):
        return {'feature_vector': self.feature_vector,
                'family': self.family,
                'breed': self.breed,
                'filename': self.filename}
    

    def show_image(self):
        plt.imshow(self.image_in_opencv_format)
        plt.show()


def create_pet_images_batch():
    dog_image_files = take_image_files_sample(250, 'dog')
    cat_image_files = take_image_files_sample(250, 'cat')

    pet_image_files = dog_image_files + cat_image_files

    pet_images = []

    print('Start processing batch of images')
    for pet_image_file in tqdm(pet_image_files):
        pet_image = PetImage(pet_image_file)

        # for some images it may not be possible to extract feature vectors
        # these images will be discarded from the batch
        if pet_image.feature_vector is not None:
            pet_image.encode_feature_vector()

            pet_images.append(pet_image.to_dict())
    
    return pet_images


if __name__=='__main__':
    pet_images = create_pet_images_batch()

    print(f'Processed {len(pet_images)} pet images')