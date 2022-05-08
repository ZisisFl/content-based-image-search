import pandas as pd
import cv2
import numpy as np
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


def compare_metrics(query_pet_image, pet_images_df, k):
    all_metrics_df = pd.DataFrame(None)

    for distance_metric_to_use in [euclidean, cityblock, minkowski, chebyshev, cosine, canberra, jaccard]:

        pet_images_df['distance'] = pet_images_df['feature_vector'].apply(lambda x: distance_metric_to_use(query_pet_image.feature_vector, x))

        k_most_similar = pet_images_df.nsmallest(k, 'distance')

        all_metrics_df = all_metrics_df.append(k_most_similar)
    
    print(all_metrics_df)


def retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric_option='cosine', verbose=False):

    if verbose:
        query_image.show_image()

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
        print('Answer set:')
        for index, item in k_most_similar_df.iterrows():
            print('{} of breed {}, image with filename {}'.format(item['family'], item['breed'], item['filename']))
    
    return k_most_similar_df


def calculate_precision(query_pet_image, retrieved_results):
    n_answer_set = len(retrieved_results.index)

    pet_family_matches = (retrieved_results['family'].values == query_pet_image.family).sum()
    pet_breed_matches = (retrieved_results['breed'].values == query_pet_image.breed).sum()

    pet_family_precision = pet_family_matches / n_answer_set
    pet_breed_precision = pet_breed_matches / n_answer_set

    print(f'Pet family precision is: {pet_family_precision}')
    print(f'Pet breed precision is: {pet_breed_precision}')


def load_database_images():
    """ Returns as a dataframe the table of pet images
    """
    postgres_db_handler = PostgresHandler()

    return postgres_db_handler.retrieve_images()


def perform_retrieval_with_pet_image():
    pet_images_df = load_database_images()

    query_image = PetImage('data/images/British_Shorthair_179.jpg')

    k = 10

    k_most_similar_df = retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric_option='cosine')

    calculate_precision(query_image, k_most_similar_df)


def main():
    pet_images_df = load_database_images()

    print(pet_images_df)

    query_image = PetImage('data/images/British_Shorthair_179.jpg')

    k = 10

    k_most_similar_df = retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric_option='cosine')

    #calculate_precision(query_image, k_most_similar_df)


if __name__=='__main__':
    main()