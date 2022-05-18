import streamlit as st
import PIL
from os import path

from image_retrieval import QueryImage
from db import PostgresHandler
from image_retrieval import retrieve_k_most_similar_images


st.set_page_config(page_title='Content-based image retrieval',
                   layout='centered')

# load images metadata dataframe from PostgreSQL
@st.experimental_singleton
def load_database_images():
    """ Returns as a dataframe the table of pet images
    """
    postgres_db_handler = PostgresHandler()

    return postgres_db_handler.retrieve_images()

try:
    pet_images_df = load_database_images()
except:
    print('Could not load table of images')
    st.error('Could not load table of images')


# description
st.title('Content-based image retrieval')
with open(path.join('assets', 'app_desc.md'), encoding='utf-8') as app_desc:
    app_description = app_desc.read()
st.markdown(app_description)

# search input
image = st.file_uploader('Choose a query image')

distance_metric = st.selectbox('Select a distance metric to perform search with',
                                ('euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard'))
    
k = st.selectbox('Select number K of similar images to retrieve',
                (5, 10, 20))

if image:
    if image.type.split('/')[0] == 'image':
        st.text('Loaded query image')
        # load PIL image
        pil_image = PIL.Image.open(image)
        # display image element
        st.image(pil_image)

        if st.button('Search'):
            # create QueryImage object
            query_image = QueryImage(pil_image, image.name)

            k_most_similar_df = retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric)

            st.header('Results')
            st.dataframe(k_most_similar_df.reset_index(drop=True))

            family_results = k_most_similar_df['family'].value_counts()

            st.text('Cats: {}'.format(family_results['cat'] if 'cat' in family_results.index else 0))
            st.text('Dogs: {}'.format(family_results['dog'] if 'dog' in family_results.index else 0))

            st.subheader('Images')
            for filename in k_most_similar_df['filename']:
                st.text(filename)
                pil_image = PIL.Image.open(path.join('data/images', filename))
                st.image(pil_image)

    else:
        st.error('You must upload an image file!')