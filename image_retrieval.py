import pandas as pd
import cv2
import PIL
import argparse
import numpy as np
from os import path
from db import PostgresHandler
from feature_generator import PetImage
#https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev, cosine, canberra, jaccard


class QueryImage:
    def __init__(self, pil_image, filename) -> None:
        self.image_in_opencv_format = self.transform_PIL_to_openCV(pil_image)
        self.feature_vector = PetImage.extract_feature_vector(self, self.image_in_opencv_format)
        self.filename = filename
        

    def transform_PIL_to_openCV(self, pil_image):
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def to_dict(self):
        return {'feature_vector': self.feature_vector,
                'filename': self.filename}


def retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric_option='cosine', verbose=False):

    if distance_metric_option == 'cosine':
        distance_metric_to_use = cosine
    elif distance_metric_option == 'euclidean':
        distance_metric_to_use = euclidean
    elif distance_metric_option == 'cityblock':
        distance_metric_to_use = cityblock
    elif distance_metric_option == 'minkowski':
        distance_metric_to_use = minkowski
    elif distance_metric_option == 'chebyshev':
        distance_metric_to_use = chebyshev
    elif distance_metric_option == 'canberra':
        distance_metric_to_use = canberra
    elif distance_metric_option == 'jaccard':
        distance_metric_to_use = jaccard
    else:
        raise ValueError(f'Distance metric {distance_metric_option} not implemented')
    
    pet_images_df['distance'] = pet_images_df['feature_vector'].apply(lambda x: distance_metric_to_use(query_image.feature_vector, x))

    k_most_similar_df = pet_images_df.nsmallest(k, 'distance')

    if verbose:
        print(f'Top {k} most similar images of {query_image.filename}')
        print('Answer set:')
        for index, item in k_most_similar_df.iterrows():
            print('{} of breed {}, image with filename {}'.format(item['family'], item['breed'], item['filename']))
    
    return k_most_similar_df


def calculate_precision(query_pet_image_family, retrieved_results, verbose=False):
    if query_pet_image_family not in ['cat', 'dog']:
        raise ValueError('Pet family can be either cat or dog')

    n_answer_set = len(retrieved_results.index)

    pet_family_matches = (retrieved_results['family'].values == query_pet_image_family).sum()

    pet_family_precision = pet_family_matches / n_answer_set

    if verbose:
        print(f'Pet family precision is: {pet_family_precision}')

    return pet_family_precision


def load_database_images():
    """ Returns as a dataframe the table of pet images
    """
    postgres_db_handler = PostgresHandler()

    return postgres_db_handler.retrieve_images()


def perform_content_based_retrieval(pet_images_df, image_filepath, k, distance_metric_option, query_pet_image_family, verbose=False):
    query_image = QueryImage(PIL.Image.open(image_filepath), image_filepath)

    k_most_similar_df = retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric_option, verbose)

    pet_family_precision = calculate_precision(query_pet_image_family, k_most_similar_df, verbose)

    # create list of image filenames after adding relative path to the images folder
    image_filenames_list = k_most_similar_df['filename'].apply(lambda x: path.join('data', 'images', x)).tolist()

    # store images collage
    create_results_collage(image_filenames_list, f'{distance_metric_option}_k{k}')

    return k_most_similar_df, pet_family_precision


def compare_metrics(image_filepath, k, query_pet_image_family):
    all_metrics_df = pd.DataFrame(None)

    pet_images_df = load_database_images()

    for distance_metric_to_use in ['euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard']:

        k_most_similar_df, pet_family_precision = perform_content_based_retrieval(pet_images_df, image_filepath, k, distance_metric_to_use, query_pet_image_family)

        print(f'For metric {distance_metric_to_use} precision is {pet_family_precision}')

        all_metrics_df = all_metrics_df.append(k_most_similar_df)


def create_results_collage(image_filenames_list, output_filename):
    pil_images = []

    # create list of PIL images
    for image in image_filenames_list:
        pil_images.append(PIL.Image.open(image))
    
    # create collage image
    collage_image = pil_grid(pil_images, 3)

    # scale down overall image size
    collage_image.resize((int(collage_image.size[0]/4), int(collage_image.size[1]/4)))

    # store output collage image
    collage_image.save(path.join('data', 'results', f'{output_filename}.jpg'))


def pil_grid(images, max_horiz=np.iinfo(int).max):
    ''' Given a list o PIL images and number of images to stach horizontally creates a PIL image collage
    '''
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * ((n_images // n_horiz) + (1 if n_images % n_horiz > 0 else 0))
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = PIL.Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Image retrieval experiments',
        add_help=True)

    parser.add_argument(
        '--action', 
        help='''Create, ingest records in or delete records from pet_images table in PostgreSQL database''', 
        type=str, 
        choices=['create', 'ingest', 'delete'], 
        required=True)

    return parser.parse_args()


if __name__=='__main__':
    #main()
    compare_metrics('data/images/basset_hound_113.jpg', 10, 'dog')
    #images = ['data/images/basset_hound_113.jpg', 'data/images/basset_hound_112.jpg']

    #create_results_collage(images, 'experiment1')