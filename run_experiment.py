#!/usr/bin/env python
# run_experiment.py - Comprehensive experiment runner for Spider dataset
import os
import sys
import json
import time
import argparse
import logging
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment(args):
    """Set up the environment for the experiment"""
    # Check if the Spider dataset path exists
    if not os.path.exists(args.spider_path):
        logger.error(f"Spider dataset path does not exist: {args.spider_path}")
        sys.exit(1)
    
    # Check for critical files
    critical_files = [
        'dev.json',
        'tables.json',
        'database'
    ]
    
    missing_files = [f for f in critical_files if not os.path.exists(os.path.join(args.spider_path, f))]
    if missing_files:
        logger.error(f"Missing critical Spider dataset files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Set SPIDER_DATASET_PATH environment variable
    os.environ['SPIDER_DATASET_PATH'] = args.spider_path
    
    # Check for required API keys
    if not os.environ.get('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY environment variable not set")
        if args.openai_api_key:
            os.environ['OPENAI_API_KEY'] = args.openai_api_key
            logger.info("Set OPENAI_API_KEY from command line argument")
        else:
            logger.error("OPENAI_API_KEY is required but not provided")
            sys.exit(1)
    
    # Return experiment directory
    return experiment_dir

def count_dev_examples(spider_path):
    """Count the number of examples in the dev set"""
    dev_file = os.path.join(spider_path, 'dev.json')
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    return len(dev_data)

def run_batch_test(args, experiment_dir):
    """Run the batch test on the Spider dataset"""
    logger.info(f"Starting batch test with limit: {args.limit}, start_index: {args.start_index}")
    
    # Build command
    cmd = [
        sys.executable,
        "run.py",
        "--mode", "batch",
        "--dataset-path", args.spider_path,
        "--dataset-type", "spider",
        "--limit", str(args.limit),
        "--start-index", str(args.start_index)
    ]
    
    if args.timeout:
        cmd.extend(["--timeout", str(args.timeout)])
    
    # Log command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run command
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            logger.error(f"Batch test failed with return code: {return_code}")
            return None
    except Exception as e:
        logger.error(f"Error running batch test: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Batch test completed in {elapsed_time:.2f} seconds")
    
    # Find result file
    results_dir = Path("outputs/spider_results")
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return None
    
    # Find the latest result file
    result_files = list(results_dir.glob("spider_results_*_final.json"))
    if not result_files:
        logger.error("No result files found")
        return None
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_result_file = result_files[0]
    
    logger.info(f"Found latest result file: {latest_result_file}")
    
    # Copy result file to experiment directory
    experiment_result_file = experiment_dir / latest_result_file.name
    shutil.copy2(latest_result_file, experiment_result_file)
    
    logger.info(f"Copied result file to: {experiment_result_file}")
    
    return experiment_result_file

def run_evaluation(args, experiment_dir, result_file):
    """Run evaluation on the batch test results"""
    if not result_file or not os.path.exists(result_file):
        logger.error(f"Result file not found: {result_file}")
        return False
    
    logger.info(f"Running evaluation on result file: {result_file}")
    
    # Basic evaluation
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    
    # Build and run command
    cmd = [
        sys.executable,
        "spider_evaluation.py",
        str(result_file),
        "--output-dir", str(evaluation_dir),
        "--create-plots",
        "--generate-report"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running evaluation: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
    # Official evaluation if enabled
    if args.official_eval:
        official_eval_dir = experiment_dir / "official_evaluation"
        official_eval_dir.mkdir(exist_ok=True)
        
        # Build and run command
        cmd = [
            sys.executable,
            "official_spider_evaluator.py",
            str(result_file),
            "--spider-path", args.spider_path,
            "--output-dir", str(official_eval_dir),
            "--eval-type", "all"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running official evaluation: {str(e)}")
            # Continue anyway, as this is just additional evaluation
        except Exception as e:
            logger.error(f"Error running official evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue anyway, as this is just additional evaluation
    
    return True

def create_experiment_report(args, experiment_dir, result_file):
    """Create a comprehensive experiment report"""
    report_file = experiment_dir / "experiment_report.md"
    
    # Load result data
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Calculate statistics
    total_queries = results.get("total_queries", 0)
    successful_queries = results.get("successful_queries", 0)
    success_rate = successful_queries / total_queries if total_queries > 0 else 0
    avg_structure_score = results.get("average_structure_score", 0)
    avg_execution_score = results.get("average_execution_score", 0)
    avg_combined_score = results.get("average_combined_score", 0)
    elapsed_time = results.get("total_elapsed_time", 0)
    
    # Create report
    report = []
    
    # Add header
    report.append(f"# Spider Dataset Experiment Report\n")
    report.append(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add experiment configuration
    report.append(f"## Experiment Configuration\n")
    report.append(f"- Spider Dataset Path: {args.spider_path}")
    report.append(f"- Limit: {args.limit}")
    report.append(f"- Start Index: {args.start_index}")
    report.append(f"- Timeout: {args.timeout} seconds")
    report.append(f"- Official Evaluation: {'Enabled' if args.official_eval else 'Disabled'}")
    
    # Add results summary
    report.append(f"\n## Results Summary\n")
    report.append(f"- Total Queries: {total_queries}")
    report.append(f"- Successful Queries: {successful_queries}")
    report.append(f"- Success Rate: {success_rate * 100:.2f}%")
    report.append(f"- Average Structure Score: {avg_structure_score:.4f}")
    report.append(f"- Average Execution Score: {avg_execution_score:.4f}")
    report.append(f"- Average Combined Score: {avg_combined_score:.4f}")
    report.append(f"- Total Elapsed Time: {elapsed_time:.2f} seconds")
    
    # Add visualizations if available
    eval_plots_dir = experiment_dir / "evaluation"
    
    if os.path.exists(eval_plots_dir / "score_distributions.png"):
        report.append(f"\n## Score Distributions\n")
        report.append(f"![Score Distributions](evaluation/score_distributions.png)")
    
    if os.path.exists(eval_plots_dir / "database_performance.png"):
        report.append(f"\n## Database Performance\n")
        report.append(f"![Database Performance](evaluation/database_performance.png)")
    
    if os.path.exists(eval_plots_dir / "error_analysis.png"):
        report.append(f"\n## Error Analysis\n")
        report.append(f"![Error Analysis](evaluation/error_analysis.png)")
    
    # Add official evaluation results if available
    official_eval_file = experiment_dir / "official_evaluation/evaluation_results.json"
    if os.path.exists(official_eval_file):
        with open(official_eval_file, 'r', encoding='utf-8') as f:
            official_results = json.load(f)
        
        report.append(f"\n## Official Spider Evaluation Results\n")
        report.append(f"- Exact Match Accuracy: {official_results.get('exact_match', 0) * 100:.2f}%")
        report.append(f"- Execution Accuracy: {official_results.get('execution_match', 0) * 100:.2f}%")
    
    # Add file references
    report.append(f"\n## Result Files\n")
    report.append(f"- Raw Results: {os.path.basename(result_file)}")
    report.append(f"- Evaluation Report: evaluation/evaluation_report.md")
    
    if args.official_eval and os.path.exists(experiment_dir / "official_evaluation/evaluation_output.txt"):
        report.append(f"- Official Evaluation Output: official_evaluation/evaluation_output.txt")
    
    # Write report to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Experiment report created: {report_file}")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Run experiments on the Spider dataset')
    
    # Required arguments
    parser.add_argument('--spider-path', type=str, required=True,
                        help='Path to the Spider dataset directory')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                        help='Directory to save experiment results')
    parser.add_argument('--limit', type=int, default=10,
                        help='Maximum number of queries to test (default: 10)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index for testing (default: 0)')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Timeout in seconds for each query (default: 600)')
    parser.add_argument('--official-eval', action='store_true',
                        help='Run official Spider evaluation script')
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key (optional, can also use environment variable)')
    
    args = parser.parse_args()
    
    # Setup environment
    experiment_dir = setup_environment(args)
    
    # Timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Count total examples
    total_examples = count_dev_examples(args.spider_path)
    logger.info(f"Total examples in Spider dev set: {total_examples}")
    
    # Validate limit and start_index
    if args.start_index >= total_examples:
        logger.error(f"Start index {args.start_index} is out of range (0-{total_examples-1})")
        sys.exit(1)
    
    if args.limit <= 0:
        logger.error(f"Limit must be greater than 0")
        sys.exit(1)
    
    if args.start_index + args.limit > total_examples:
        old_limit = args.limit
        args.limit = total_examples - args.start_index
        logger.warning(f"Limiting to {args.limit} examples (requested {old_limit}, but only {args.limit} available)")
    
    # Run batch test
    result_file = run_batch_test(args, experiment_dir)
    
    if not result_file:
        logger.error("Batch test failed or no results found")
        sys.exit(1)
    
    # Run evaluation
    success = run_evaluation(args, experiment_dir, result_file)
    
    if not success:
        logger.error("Evaluation failed")
        sys.exit(1)
    
    # Create experiment report
    report_file = create_experiment_report(args, experiment_dir, result_file)
    
    logger.info(f"Experiment completed successfully")
    logger.info(f"Report file: {report_file}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    main()