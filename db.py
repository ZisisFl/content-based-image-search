import pickle
import argparse
from dotenv import load_dotenv
from os import getenv
from sqlalchemy import create_engine, MetaData, Table, Column, insert
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, BYTEA
from pandas import read_sql


class PostgresHandler:
    """ Handler for connection with PostgreSQL backend database

    This class provides methods to ingest images in the database 
    and retrieve them in the form of pandas dataframes
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
                            Column('family', VARCHAR(3)),
                            Column('breed', VARCHAR),
                            Column('filename', VARCHAR)
                            )
        
        # if table doesn't exist then create it else nothing due to checkfirst 
        try:
            metadata.create_all(bind=self.engine, checkfirst=True, tables=[target_table])            
        except Exception as e:
            raise e

        return target_table
    

    def ingest_images(self):
        from feature_generator import create_pet_images_batch
        # create batch of pet images to store in database
        pet_images = create_pet_images_batch()

        insert_statement = insert(self.target_table).values(pet_images)
        
        try:
            print(f'Ingesting {len(pet_images)} image records in pet_images table')
            self.engine.execute(insert_statement)
        except Exception as e:
            raise e
        
        print('Ingested pet images in table successfully')
    

    def delete_images(self):
        delete_statement = self.target_table.delete()

        try:
            self.engine.execute(delete_statement)
        except Exception as e:
            raise e
        
        print('Deleted all records of pet_images table successfully')
    

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


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description='Content-based image search Database handler CLI',
        add_help=True)

    parser.add_argument(
        '--action', 
        help='''Create, ingest records in or delete records from pet_images table in PostgreSQL database''', 
        type=str, 
        choices=['create', 'ingest', 'delete'], 
        required=True)

    return parser.parse_args()


def main():
    args = parse_command_line_arguments()

    postgres_db_handler = PostgresHandler()
    
    if args.action == 'create':
        postgres_db_handler.create_images_table()
    elif args.action == 'ingest':
        postgres_db_handler.ingest_images()
    elif args.action == 'delete':
        postgres_db_handler.delete_images()


if __name__=='__main__':
    # # initialize connection to postgresql with db handler
    # postgres_db_handler = PostgresHandler()

    # # create instance of pet images target table
    # # if table doesn't exist in database it will create it
    # postgres_db_handler.create_images_table()

    # # ingest sample of images in the database
    # postgres_db_handler.ingest_images()
    main()