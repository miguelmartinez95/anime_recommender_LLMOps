from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

load_dotenv()

logger = get_logger(__name__)

def main():
    try: 
        logger.info("Starting to buid pipeline....")

        loader = AnimeDataLoader("data/anime_with_synopsis.csv", "data/procesed_anyme.csv")
        processed_csv = loader.load_and_process()

        logger.info("Data loaded and processed...")

        vector_builder = VectorStoreBuilder(processed_csv)
        vector_builder.build_and_save_vectorstore()

        logger.info("VectorStore built sucessfully.....")
    except Exception as e:
        logger.error(f"Error building the vector store: {e}")
        raise CustomException("Error when building the VectorStore", e)

if __name__ == "__main__":
    main()