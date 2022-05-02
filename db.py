import pickle
from dotenv import load_dotenv
from os import getenv
from sqlalchemy import create_engine, MetaData, Table, Column, insert
from sqlalchemy.dialects.postgresql import TEXT, INTEGER, BYTEA
from pandas import read_sql

from feature_generator import create_pet_images_batch


class PostgresHandler:
    """
    """

    def __init__(self) -> None:
        self.engine = create_engine(self._construct_connection_string(), pool_pre_ping=True)
        self.target_table = self.create_images_table()


    def _construct_connection_string(self):
        load_dotenv()
        
        host = getenv('DB_HOST')
        port = getenv('DB_PORT')
        db = getenv('DB_NAME')
        user = getenv('DB_USER')
        secret = getenv('DB_SECRET')

        return f'postgresql://{user}:{secret}@{host}:{port}/{db}'


    def create_images_table(self):
        metadata = MetaData(self.engine)

        target_table = Table('pet_images',
                            metadata,
                            Column('id', INTEGER, primary_key=True),
                            Column('feature_vector', BYTEA),
                            Column('family', TEXT),
                            Column('breed', TEXT),
                            Column('filename', TEXT)
                            )
        
        # if table doesn't exist then create it else nothing due to checkfirst 
        try:
            metadata.create_all(bind=self.engine, checkfirst=True, tables=[target_table])            
        except Exception as e:
            raise e

        return target_table
    

    def ingest_images(self):
        # create batch of pet images to store in database
        pet_images = create_pet_images_batch()
        
        try:
            print('Ingesting pet images in table')
            self.engine.execute(insert(self.target_table).values(pet_images))
        except Exception as e:
            raise e
        
        print('Ingested pet images in table successfully')
    

    def retrieve_images(self):
        try:
            # retrieve pet_images table 
            query = 'SELECT * FROM pet_images'
            df = read_sql(query, self.engine)
            
        except Exception as e:
            raise e

        # decode feature vector
        df['feature_vector'] = df['feature_vector'].apply(lambda x: pickle.loads(x))
        
        return df


if __name__=='__main__':
    # initialize connection to postgresql with db handler
    postgres_db_handler = PostgresHandler()

    # create instance of pet images target table
    # if table doesn't exist in database it will create it
    postgres_db_handler.create_images_table()

    # ingest sample of images in the database
    postgres_db_handler.ingest_images()