import pandas as pd
import PIL
import numpy as np
from os import path

from image_retrieval import QueryImage, retrieve_k_most_similar_images, load_database_images

def calculate_precision(query_pet_image_family, query_pet_image_breed, retrieved_results, verbose=False):
    if query_pet_image_family not in ['cat', 'dog']:
        raise ValueError('Pet family can be either cat or dog')

    n_answer_set = len(retrieved_results.index)

    pet_family_matches = (retrieved_results['family'].values == query_pet_image_family).sum()
    pet_breed_matches = (retrieved_results['breed'].values == query_pet_image_breed).sum()

    pet_family_precision = pet_family_matches / n_answer_set
    pet_breed_precision = pet_breed_matches / n_answer_set

    if verbose:
        print(f'Pet family precision is: {pet_family_precision}')
        print(f'Pet breed precision is: {pet_breed_precision}')

    return pet_family_precision, pet_breed_precision


def perform_content_based_retrieval(pet_images_df, image_filepath, k, distance_metric_option, query_pet_image_family, query_pet_image_breed, verbose=False):
    query_image = QueryImage(PIL.Image.open(image_filepath), image_filepath)

    k_most_similar_df = retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric_option, verbose)

    pet_family_precision, pet_breed_precision = calculate_precision(query_pet_image_family, query_pet_image_breed, k_most_similar_df, verbose)

    # create list of image filenames after adding relative path to the images folder
    image_filenames_list = k_most_similar_df['filename'].apply(lambda x: path.join('data', 'images', x)).tolist()

    # store images collage
    create_results_collage(image_filenames_list, f'''{query_image.filename.split('.')[0]}_{distance_metric_option}_k{k}''')

    return k_most_similar_df, pet_family_precision, pet_breed_precision


def compare_metrics(image_filepath, k, query_pet_image_family, query_pet_image_breed):
    all_metrics_df = pd.DataFrame(None)

    pet_images_df = load_database_images()

    for distance_metric_to_use in ['euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard']:

        k_most_similar_df, pet_family_precision, pet_breed_precision = perform_content_based_retrieval(pet_images_df, image_filepath, k, distance_metric_to_use, query_pet_image_family, query_pet_image_breed)

        print(f'For metric {distance_metric_to_use} pet family precision is {pet_family_precision}')
        print(f'For metric {distance_metric_to_use} pet breed precision is {pet_breed_precision}')

        # concat all results in a single dataframe
        all_metrics_df = pd.concat([all_metrics_df, k_most_similar_df])


def create_results_collage(image_filenames_list, output_filename):
    pil_images = []

    # max size that the biggest dimension of an image can have
    max_size = 300

    # create list of PIL images
    for image_file in image_filenames_list:
        image = PIL.Image.open(image_file)
        
        size_of_biggest_image_dimension = max(image.size)

        # if size_of_biggest_image_dimension is bigger than max size requested scale down image
        if max_size < size_of_biggest_image_dimension:
            # find required scale down factor
            scale_down_factor = size_of_biggest_image_dimension/max_size

            # scale down image
            image = image.resize((int(image.size[0]/scale_down_factor), int(image.size[1]/scale_down_factor)))

        pil_images.append(image)
    
    # create collage image
    collage_image = pil_grid(pil_images, 3)

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


def run_experiments():
    for k in [5,10,20]:
        print(f'With k={k}:')

        image_file = 'basset_hound_113.jpg'
        print(f'For image {image_file}')
        compare_metrics(path.join('data', 'sample_query_images', image_file), k, 'dog', 'basset_hound')
        print('\n')

        image_file = 'Bengal_29.jpg'
        print(f'For image {image_file}')
        compare_metrics(path.join('data', 'sample_query_images', image_file), k, 'cat', 'bengal')
        print('\n')

        image_file = 'out_of_dataset_dog.jpeg'
        print(f'For image {image_file}')
        compare_metrics(path.join('data', 'sample_query_images', image_file), k, 'dog', 'pug')
        print('\n')


if __name__=='__main__':
    run_experiments()