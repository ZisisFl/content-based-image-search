from os import path
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev, cosine, canberra, jaccard

from db import PostgresHandler
from feature_generator import PetImage


class QueryImage:
    def __init__(self, pil_image, filename) -> None:
        self.image_in_pil_format = pil_image
        self.feature_vector = PetImage.extract_feature_vector(self, self.image_in_pil_format)
        self.filename = path.split(filename)[1]

    
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


def load_database_images():
    """ Returns as a dataframe the table of pet images
    """
    postgres_db_handler = PostgresHandler()

    return postgres_db_handler.retrieve_images()