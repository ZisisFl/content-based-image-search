# Content-based image retrieval
This is a simple project that implements a content-based image retrieval engine using PostgreSQL as storage backend and Python for the application logic. It uses [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16) to extract features from images.

## Dataset
You can find the original (Oxford-IIIT Pet Dataset) dataset used in this project in the following [link](https://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset consists of roughly 7000 dog and cat images annotated with their respective breeds.

## Requirments
### Python 
You will need to create a Python virtual environment and install the packages found in requirements.txt file.

Create and activate Python virtual env using anaconda
```sh
conda create --name image_search_env python=3.8
conda activate image_search_env
```

Install base required python packages
```sh
pip install -r base_requirements.txt
```

File `base_requirements.txt` contains all packages you need to install for this project except tensorflow related ones. This is because this project was developed in a Apple M1 machine and requires different process for install tensorflow.

For Apple M1 do the following:
```sh
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
```

In other systems the following command should be enough (haven't tried tho):
```sh
pip install tensorflow
```

### PostgreSQL
In order to use this application you need to set up a connection with a PostgreSQL database to create and populate a table named `pet_images`.
To set up connection with PostgreSQL you need to create a .env file with the following format.

```sh
DB_HOST=localhost
DB_PORT=5432
DB_NAME=image_search
DB_USER=postgres
DB_SECRET=postgres
```

## Use the Application
Follow the steps described in the next sections to setup and start the application.
### Setting up dataset of images
1. Download dataset of images using this [link](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz) (clicking the link will start the downloading of a GZ file named images.tar.gz containing `images` folder).
2. Unzip the images.tar.gz file to extract images folder
3. Place images folder under data directory of the project

### Setting up PostgreSQL database

1. Execute the following command to create `pet_images` table that will host links to pet image files, metadata and feature vectors:
    ```python
    python db.py --action=create
    ```
2. Execute the following command to ingest a sample of roughly 250 cat and 250 dog image records in `pet_images` table of PostgreSQL database. Images are randomly selected using take_image_files_sample from image_sampling.py module
    ```python
    python db.py --action=ingest
    ```
    
[Optional] Execute the following command to delete any image records in `pet_images` table:
```python
python db.py --action=delete
```

### Start streamlit web app
In order to start the Python web application activate the Python virtual environment as described in section Requirements and execute the following command 
```python
streamlit run app.py --server.port=5555
```

Application will open up in broswer
![app_preview_image](/assets/app_preview.png)