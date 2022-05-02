import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from images_sampling import take_image_files_sample


class PetImage:
    
    def __init__(self, filename) -> None:
        self.feature_vector = self.extract_feature_vector(filename)
        self.family = self.extract_pet_family(filename)
        self.breed = self.extract_pet_breed(filename)
        self.filename = filename
    

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
    

    def extract_feature_vector(self, image_filename, vector_size=32):
        # TODO images like data/images/Egyptian_Mau_177.jpg fail to load
        image = cv2.imread(image_filename, 0)

        try:
            alg = cv2.SIFT_create()
            # Dinding image keypoints
            kps = alg.detect(image)

            # Getting first 32 of them. 
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
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print(f'Error: {e}')
            return None

        return dsc
    

    def to_dict(self):
        return {'feature_vector': self.feature_vector,
                'family': self.family,
                'breed': self.breed,
                'filename': self.filename}
    

    def show_image(self):
        image = cv2.imread(self.filename)
        plt.imshow(image)
        plt.show()


def create_pet_images_batch():
    dog_image_files = take_image_files_sample(250, 'dog')
    cat_image_files = take_image_files_sample(250, 'cat')

    pet_image_files = dog_image_files + cat_image_files

    pet_images = []

    for pet_image_file in pet_image_files:
        print(pet_image_file)
        pet_image = PetImage(pet_image_file)

        pet_image.encode_feature_vector()

        pet_images.append(pet_image.to_dict())
    
    return pet_images


if __name__=='__main__':
    pet_images = create_pet_images_batch()
    print(pet_images)