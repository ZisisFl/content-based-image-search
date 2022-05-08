import streamlit as st
import PIL
from os import path
from image_retrieval import QueryImage
from db import PostgresHandler
from image_retrieval import retrieve_k_most_similar_images


st.set_page_config(page_title="Content-based image retrieval",
                   layout="centered")


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

st.title('Content-based image retrieval')
st.markdown('''This is a simple project that implements a content-based image retrieval engine using PostgreSQL as storage backend 
and Python for the application logic. There is a database of 500 pet images 250 of them are classified as dogs and 250 of them 
as cats along with their breeds. 

With this application you can upload your own image and search for similar images given the number K of images to retrieve and the 
distance metric to use for the ranking ''')

image = st.file_uploader('Choose a query image')

distance_metric = st.selectbox('Select a distance metric to perform search with',
                                ('euclidean', 'cityblock', 'minkowski', 'chebyshev', 'cosine', 'canberra', 'jaccard'))
    
k = st.selectbox('Select number K of similar images to retrieve',
                (5, 10, 20))

if image:
    st.text('Loaded query image')
    # load PIL image
    pil_image = PIL.Image.open(image)
    # display image element
    st.image(pil_image)


if st.button('Search'):
    
    if image:
        # create QueryImage object
        query_image = QueryImage(pil_image, image.name)

        k_most_similar_df = retrieve_k_most_similar_images(query_image, pet_images_df, k, distance_metric)

        st.header('Results')
        st.dataframe(k_most_similar_df.reset_index(drop=True))

        #if st.button('Show images'):

        for filename in k_most_similar_df['filename']:
            st.text(filename)
            pil_image = PIL.Image.open(path.join('data/images', filename))
            st.image(pil_image)
        
        # maybe change with collage https://pillow.readthedocs.io/en/stable/reference/Image.html
    else:
        st.warning('You must select a query image first!')