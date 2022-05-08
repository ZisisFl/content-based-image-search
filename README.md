# Content-based image search
This is a simple project that implements a content-based image retrieval engine using PostgreSQL as storage backend and Python for the application logic.

## Dataset
You can find the original (Oxford-IIIT Pet Dataset) dataset used in this project in the following [link](https://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset consists of roughly 7000 dog and cat images annotated in different breeds.

## Requirments
### Python 
You will need to create a Python virtual environment and install the packages found in requirements.txt file.

Create and activate Python virtual env using anaconda
```sh
conda create --name image_search_env python=3.8
conda activate image_search_env
```

Install required python packages
```sh
pip install -r requirements.txt
```

### PostgreSQL
In order to use this application you need to set up a connection with a PostgreSQL database to create and populate a table named pet_images records.
To set up connection with PostgreSQL you need to create a .env file with the following format.

```sh
DB_HOST=localhost
DB_PORT=5432
DB_NAME=image_search
DB_USER=postgres
DB_SECRET=postgres
```