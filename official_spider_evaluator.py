#!/usr/bin/env python
# official_spider_evaluator.py - Bridge to Spider's official evaluation
import os
import sys
import json
import argparse
import subprocess
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_spider_eval_script(spider_path):
    """Find the official Spider evaluation script"""
    # Try common locations
    potential_paths = [
        os.path.join(spider_path, 'evaluation.py'),
        os.path.join(spider_path, 'eval', 'evaluation.py'),
        os.path.join(spider_path, 'evaluation', 'evaluation.py')
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    return None

def prepare_predictions_file(results, output_path):
    """Prepare a predictions file in the format expected by Spider evaluation"""
    predictions = []
    
    for result in results.get("results", []):
        if "generated_sql" in result and result["generated_sql"]:
            predictions.append({
                "query": result["generated_sql"],
                "question_id": result.get("index", 0),
                "db_id": result.get("db_id", "")
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    return len(predictions)

def prepare_gold_file(results, output_path):
    """Prepare a gold file in the format expected by Spider evaluation"""
    gold_data = []
    
    for result in results.get("results", []):
        if "gold_sql" in result and result["gold_sql"]:
            gold_data.append({
                "query": result["gold_sql"],
                "question": result.get("question", ""),
                "question_id": result.get("index", 0),
                "db_id": result.get("db_id", "")
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, ensure_ascii=False, indent=2)
    
    return len(gold_data)

def run_official_evaluation(gold_file, pred_file, db_dir, etype, eval_script=None):
    """Run the official Spider evaluation script"""
    try:
        if eval_script and os.path.exists(eval_script):
            # Use the provided script
            cmd = [sys.executable, eval_script, gold_file, pred_file, db_dir, etype]
        else:
            # Try using the evaluation script installed in the Python package
            cmd = [sys.executable, "-m", "spider.evaluation", gold_file, pred_file, db_dir, etype]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return {
            "success": True,
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "output": e.stdout if hasattr(e, 'stdout') else "",
            "error": e.stderr if hasattr(e, 'stderr') else str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }

def parse_evaluation_results(output):
    """Parse the output of the Spider evaluation script"""
    results = {
        "exact_match": 0.0,
        "execution_match": 0.0
    }
    
    if not output:
        return results
    
    # Extract exact match score
    exact_match = re.search(r'Exact matching accuracy: (\d+\.\d+)%', output)
    if exact_match:
        results["exact_match"] = float(exact_match.group(1)) / 100.0
    
    # Extract execution match score
    execution_match = re.search(r'Execution accuracy: (\d+\.\d+)%', output)
    if execution_match:
        results["execution_match"] = float(execution_match.group(1)) / 100.0
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate SQL using Spider official evaluation')
    parser.add_argument('results_file', type=str, help='Path to the results JSON file')
    parser.add_argument('--spider-path', type=str, required=True,
                        help='Path to the Spider dataset directory')
    parser.add_argument('--eval-script', type=str, default=None,
                        help='Path to the Spider evaluation script (optional)')
    parser.add_argument('--output-dir', type=str, default='outputs/spider_official_eval',
                        help='Directory to save evaluation results')
    parser.add_argument('--eval-type', type=str, default='all',
                        choices=['all', 'exec', 'match'],
                        help='Evaluation type: exec (execution), match (exact match), or all')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(args.results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Find evaluation script
    eval_script = args.eval_script
    if not eval_script:
        eval_script = find_spider_eval_script(args.spider_path)
        if not eval_script:
            logger.warning("Spider evaluation script not found. Will try to use the package.")
    
    # Create temporary files for predictions and gold data
    pred_file = output_dir / 'predictions.json'
    gold_file = output_dir / 'gold.json'
    
    # Prepare prediction and gold files
    num_pred = prepare_predictions_file(results, pred_file)
    num_gold = prepare_gold_file(results, gold_file)
    
    logger.info(f"Prepared {num_pred} predictions and {num_gold} gold data points")
    
    # Determine database directory
    db_dir = os.path.join(args.spider_path, 'database')
    if not os.path.exists(db_dir):
        logger.error(f"Database directory not found: {db_dir}")
        sys.exit(1)
    
    # Run evaluation
    logger.info(f"Running official Spider evaluation with type: {args.eval_type}")
    eval_results = run_official_evaluation(
        str(gold_file),
        str(pred_file),
        db_dir,
        args.eval_type,
        eval_script
    )
    
    if eval_results["success"]:
        logger.info("Evaluation completed successfully")
        
        # Save raw output
        with open(output_dir / 'evaluation_output.txt', 'w', encoding='utf-8') as f:
            f.write(eval_results["output"])
        
        # Parse results
        parsed_results = parse_evaluation_results(eval_results["output"])
        
        # Save parsed results
        with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(parsed_results, f, ensure_ascii=False, indent=2)
        
        # Print results
        print("\nOfficial Spider Evaluation Results:")
        print(f"Exact Match Accuracy: {parsed_results['exact_match'] * 100:.2f}%")
        print(f"Execution Accuracy: {parsed_results['execution_match'] * 100:.2f}%")
        
        print("\nFull evaluation output saved to:", output_dir / 'evaluation_output.txt')
    else:
        logger.error("Evaluation failed")
        logger.error(eval_results["error"])
        
        with open(output_dir / 'evaluation_error.txt', 'w', encoding='utf-8') as f:
            f.write(eval_results["error"])
        
        print("\nEvaluation failed. See error log:", output_dir / 'evaluation_error.txt')

if __name__ == "__main__":
    import re  # Import here to avoid circular imports
    main()