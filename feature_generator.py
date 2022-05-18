import numpy as np
import pickle
from tqdm import tqdm
from os import path
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from images_sampling import take_image_files_sample

print('Loading VGG model')
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
print('Loaded VGG model')

class PetImage:
    
    def __init__(self, filename) -> None:
        self.image_in_pil_format = Image.open(filename)
        self.feature_vector = self.extract_feature_vector(self.image_in_pil_format)
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
    

    def extract_feature_vector(self, image):
        try:
            # VGG must take a 224x224 img as an input
            image = image.resize((224, 224))
            image = image.convert('RGB')

            # To np.array. Height x Width x Channel. dtype=float32
            x = img_to_array(image)

            # (H, W, C)->(1, H, W, C), where the first elem is the number of img
            x = np.expand_dims(x, axis=0)

            # Subtracting avg values for each pixel
            x = preprocess_input(x)  

            feature = model.predict(x)[0]  # (1, 4096) -> (4096, )
        except Exception as e:
            print(f'Could not extract features from image due to error: {e}')
            return None
            
        return feature / np.linalg.norm(feature)  # Normalize
    

    def to_dict(self):
        return {'feature_vector': self.feature_vector,
                'family': self.family,
                'breed': self.breed,
                'filename': self.filename}
    

    def show_image(self):
        plt.imshow(self.image_in_pil_format)
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