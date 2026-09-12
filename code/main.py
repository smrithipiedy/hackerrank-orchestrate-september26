import logging
from code.ingestion import DataIngestor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting ApexOptima Finance - Phase 1 Ingestion Test")
    try:
        ingestor = DataIngestor()
        ingestor.load_all()
        normalized_events = ingestor.normalize_events()
        logger.info(f"Successfully normalized {len(normalized_events)} events.")
        logger.info("Phase 1 Ingestion verified successfully.")
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
