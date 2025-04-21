#!/usr/bin/env python
# run.py - Optimized version for Spider dataset evaluation
import sys
import os
import warnings
import logging
import argparse
import traceback
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logging.getLogger('chromadb').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_log.txt"),
        logging.StreamHandler()
    ]
)

# Get project root directory
ROOT_DIR = Path(__file__).parent

# Load .env file
load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Add source code directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def setup_environment():
    """Setup environment variables and check prerequisites"""
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please add them to your .env file")
        sys.exit(1)
    
    # Get Spider dataset path
    spider_dir = os.getenv('SPIDER_DATASET_PATH')
    if spider_dir and os.path.exists(spider_dir):
        logger.info(f'Spider dataset path found: {spider_dir}')
        
        # Verify critical files exist
        critical_files = [
            'dev.json',
            'tables.json',
            'database'
        ]
        
        missing_files = [f for f in critical_files if not os.path.exists(os.path.join(spider_dir, f))]
        if missing_files:
            logger.warning(f"Some critical Spider dataset files are missing: {', '.join(missing_files)}")
            logger.warning("This may affect the evaluation results")
    else:
        logger.warning('Spider dataset path not found or invalid, will use default query')
        spider_dir = None
    
    return spider_dir

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run query processing flow with improved Spider dataset support')
    
    # Main arguments
    parser.add_argument('--dataset-path', type=str, help='Dataset path (overrides environment variable)')
    parser.add_argument('--dataset-type', type=str, default='spider', 
                        choices=['spider', 'custom', 'default'], 
                        help='Dataset type (spider, custom, or default)')
    parser.add_argument('--mode', type=str, default='single', 
                        choices=['single', 'batch'], 
                        help='Run mode: single query or batch test')
    
    # Batch mode arguments
    parser.add_argument('--limit', type=int, default=10, 
                        help='Maximum number of queries for batch testing')
    parser.add_argument('--start-index', type=int, default=0, 
                        help='Starting index for batch testing')
                        
    # Advanced options
    parser.add_argument('--timeout', type=int, default=600, 
                        help='Timeout in seconds for each query')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip SQL evaluation (faster)')
    
    return parser.parse_args()

def main():
    """Main function to run the query processing flow"""
    try:
        # Setup environment
        spider_dir = setup_environment()
        
        # Parse arguments
        args = parse_arguments()
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Override dataset path if provided
        dataset_path = args.dataset_path or spider_dir
        
        # Run in specified mode
        if args.mode == 'single':
            # Import here to avoid circular imports
            from multi.main import kickoff
            
            logger.info(f"Running single query with dataset type: {args.dataset_type}")
            result = kickoff(dataset_path=dataset_path, dataset_type=args.dataset_type)
            
            # Print results
            if hasattr(result, 'state') and hasattr(result.state, 'sql_query'):
                sql_query = result.state.sql_query
                print("\nGenerated SQL Query:")
                print("=" * 80)
                print(sql_query)
                print("=" * 80)
                
                # Print evaluation if gold SQL is available
                if hasattr(result.state, 'gold_sql') and result.state.gold_sql:
                    from multi.main import evaluate_sql
                    
                    if not args.skip_evaluation:
                        # Construct database path
                        db_path = None
                        if dataset_path and result.state.db_id:
                            potential_db_path = os.path.join(
                                dataset_path, 'database', result.state.db_id, f"{result.state.db_id}.sqlite"
                            )
                            if os.path.exists(potential_db_path):
                                db_path = potential_db_path
                        
                        # Evaluate SQL
                        evaluation = evaluate_sql(sql_query, result.state.gold_sql, db_path)
                        
                        print("\nSQL Evaluation:")
                        print(f"Structure Score: {evaluation['structure_score']:.2f}")
                        print(f"Execution Score: {evaluation['execution_score']:.2f}")
                        print(f"Combined Score: {evaluation['combined_score']:.2f}")
                        print("=" * 80)
        else:
            # Import here to avoid circular imports
            from multi.main import batch_test_spider
            
            logger.info(f"Running batch test with limit: {args.limit}, start index: {args.start_index}")
            batch_test_spider(
                spider_dataset_path=dataset_path, 
                limit=args.limit, 
                start_index=args.start_index
            )
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()