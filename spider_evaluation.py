#!/usr/bin/env python
# spider_evaluation.py - Utility script for evaluating Spider dataset results
import os
import json
import argparse
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

def load_results(results_file):
    """Load results from a JSON file"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_summary(results):
    """Extract summary statistics from results"""
    summary = {
        "timestamp": results.get("timestamp", "Unknown"),
        "total_queries": results.get("total_queries", 0),
        "successful_queries": results.get("successful_queries", 0),
        "average_structure_score": results.get("average_structure_score", 0),
        "average_execution_score": results.get("average_execution_score", 0),
        "average_combined_score": results.get("average_combined_score", 0),
        "total_elapsed_time": results.get("total_elapsed_time", 0)
    }
    
    # Calculate success rate
    if summary["total_queries"] > 0:
        summary["success_rate"] = summary["successful_queries"] / summary["total_queries"]
    else:
        summary["success_rate"] = 0
    
    # Calculate average time per query
    if summary["successful_queries"] > 0:
        summary["average_time_per_query"] = summary["total_elapsed_time"] / summary["successful_queries"]
    else:
        summary["average_time_per_query"] = 0
    
    return summary

def generate_score_distribution(results):
    """Generate score distribution data"""
    structure_scores = []
    execution_scores = []
    combined_scores = []
    
    for result in results.get("results", []):
        if "evaluation" in result and result["evaluation"].get("status") == "success":
            structure_scores.append(result["evaluation"].get("structure_score", 0))
            execution_scores.append(result["evaluation"].get("execution_score", 0))
            combined_scores.append(result["evaluation"].get("combined_score", 0))
    
    return {
        "structure_scores": structure_scores,
        "execution_scores": execution_scores,
        "combined_scores": combined_scores
    }

def analyze_errors(results):
    """Analyze errors in results"""
    errors = []
    
    for result in results.get("results", []):
        if result.get("error"):
            errors.append({
                "index": result.get("index"),
                "db_id": result.get("db_id"),
                "question": result.get("question"),
                "error": result.get("error")
            })
    
    # Categorize errors
    error_categories = {}
    for error in errors:
        # Extract error type (first part of error message)
        error_msg = error["error"]
        if ":" in error_msg:
            error_type = error_msg.split(":")[0].strip()
        else:
            error_type = "Unknown"
        
        if error_type not in error_categories:
            error_categories[error_type] = 0
        error_categories[error_type] += 1
    
    return {
        "total_errors": len(errors),
        "error_details": errors,
        "error_categories": error_categories
    }

def analyze_database_performance(results):
    """Analyze performance by database"""
    db_performance = {}
    
    for result in results.get("results", []):
        db_id = result.get("db_id")
        
        if db_id not in db_performance:
            db_performance[db_id] = {
                "count": 0,
                "successful": 0,
                "structure_score_sum": 0,
                "execution_score_sum": 0,
                "combined_score_sum": 0,
                "time_sum": 0
            }
        
        db_performance[db_id]["count"] += 1
        
        if "evaluation" in result and result["evaluation"].get("status") == "success":
            db_performance[db_id]["successful"] += 1
            db_performance[db_id]["structure_score_sum"] += result["evaluation"].get("structure_score", 0)
            db_performance[db_id]["execution_score_sum"] += result["evaluation"].get("execution_score", 0)
            db_performance[db_id]["combined_score_sum"] += result["evaluation"].get("combined_score", 0)
            
        if "elapsed_time" in result:
            db_performance[db_id]["time_sum"] += result["elapsed_time"]
    
    # Calculate averages
    for db_id, data in db_performance.items():
        if data["successful"] > 0:
            data["avg_structure_score"] = data["structure_score_sum"] / data["successful"]
            data["avg_execution_score"] = data["execution_score_sum"] / data["successful"]
            data["avg_combined_score"] = data["combined_score_sum"] / data["successful"]
        else:
            data["avg_structure_score"] = 0
            data["avg_execution_score"] = 0
            data["avg_combined_score"] = 0
            
        if data["count"] > 0:
            data["success_rate"] = data["successful"] / data["count"]
            data["avg_time"] = data["time_sum"] / data["count"]
        else:
            data["success_rate"] = 0
            data["avg_time"] = 0
    
    return db_performance

def create_plots(results, output_dir):
    """Create visualization plots from results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    scores = generate_score_distribution(results)
    
    # Create score distribution histograms
    plt.figure(figsize=(12, 8))
    
    # Structure scores
    plt.subplot(3, 1, 1)
    plt.hist(scores["structure_scores"], bins=10, range=(0, 1), alpha=0.7, color='blue')
    plt.title('Structure Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Execution scores
    plt.subplot(3, 1, 2)
    plt.hist(scores["execution_scores"], bins=10, range=(0, 1), alpha=0.7, color='green')
    plt.title('Execution Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Combined scores
    plt.subplot(3, 1, 3)
    plt.hist(scores["combined_scores"], bins=10, range=(0, 1), alpha=0.7, color='red')
    plt.title('Combined Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png')
    
    # Create database performance visualization
    db_performance = analyze_database_performance(results)
    
    # Sort databases by combined score
    sorted_dbs = sorted(db_performance.items(), 
                        key=lambda x: x[1]["avg_combined_score"], 
                        reverse=True)
    
    # Limit to top 15 for visibility
    top_dbs = sorted_dbs[:15]
    
    plt.figure(figsize=(14, 10))
    
    # Extract data for plotting
    db_ids = [db[0] for db in top_dbs]
    structure_scores = [db[1]["avg_structure_score"] for db in top_dbs]
    execution_scores = [db[1]["avg_execution_score"] for db in top_dbs]
    combined_scores = [db[1]["avg_combined_score"] for db in top_dbs]
    
    # Create bar positions
    x = np.arange(len(db_ids))
    width = 0.25
    
    # Create grouped bar chart
    plt.bar(x - width, structure_scores, width, label='Structure Score', color='blue', alpha=0.7)
    plt.bar(x, execution_scores, width, label='Execution Score', color='green', alpha=0.7)
    plt.bar(x + width, combined_scores, width, label='Combined Score', color='red', alpha=0.7)
    
    plt.xlabel('Database ID')
    plt.ylabel('Average Score')
    plt.title('Performance by Database (Top 15)')
    plt.xticks(x, db_ids, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'database_performance.png')
    
    # Create error analysis pie chart
    error_analysis = analyze_errors(results)
    
    if error_analysis["total_errors"] > 0:
        plt.figure(figsize=(10, 8))
        
        # Get error categories and counts
        categories = list(error_analysis["error_categories"].keys())
        counts = list(error_analysis["error_categories"].values())
        
        # Create pie chart
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Error Categories')
        
        plt.savefig(output_dir / 'error_analysis.png')
    
    print(f"Plots saved to {output_dir}")

def generate_report(results, output_dir):
    """Generate a detailed report from results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    summary = extract_summary(results)
    error_analysis = analyze_errors(results)
    db_performance = analyze_database_performance(results)
    
    # Create report
    report = []
    
    # Add header
    report.append(f"# Spider Dataset Evaluation Report\n")
    report.append(f"Generated: {summary['timestamp']}\n")
    
    # Add summary
    report.append(f"## Summary\n")
    report.append(f"- Total Queries: {summary['total_queries']}")
    report.append(f"- Successful Queries: {summary['successful_queries']}")
    report.append(f"- Success Rate: {summary['success_rate'] * 100:.2f}%")
    report.append(f"- Average Structure Score: {summary['average_structure_score']:.4f}")
    report.append(f"- Average Execution Score: {summary['average_execution_score']:.4f}")
    report.append(f"- Average Combined Score: {summary['average_combined_score']:.4f}")
    report.append(f"- Total Elapsed Time: {summary['total_elapsed_time']:.2f} seconds")
    report.append(f"- Average Time per Query: {summary['average_time_per_query']:.2f} seconds\n")
    
    # Add error analysis
    report.append(f"## Error Analysis\n")
    report.append(f"- Total Errors: {error_analysis['total_errors']}")
    report.append(f"- Error Categories:")
    
    for category, count in error_analysis["error_categories"].items():
        report.append(f"  - {category}: {count}")
    
    report.append("\n")
    
    # Add database performance
    report.append(f"## Database Performance\n")
    
    # Create a table of database performance
    db_data = []
    for db_id, data in db_performance.items():
        db_data.append([
            db_id,
            data["count"],
            f"{data['success_rate'] * 100:.2f}%",
            f"{data['avg_structure_score']:.4f}",
            f"{data['avg_execution_score']:.4f}",
            f"{data['avg_combined_score']:.4f}",
            f"{data['avg_time']:.2f}"
        ])
    
    # Sort by combined score
    db_data.sort(key=lambda x: float(x[5].replace('%', '')), reverse=True)
    
    # Create table
    db_table = tabulate(
        db_data,
        headers=["Database", "Queries", "Success Rate", "Structure Score", "Execution Score", "Combined Score", "Avg Time (s)"],
        tablefmt="pipe"
    )
    
    report.append(db_table)
    report.append("\n")
    
    # Add top/bottom performing queries
    report.append(f"## Top Performing Queries\n")
    
    # Get successful queries with evaluation
    successful_queries = []
    for result in results.get("results", []):
        if "evaluation" in result and result["evaluation"].get("status") == "success":
            successful_queries.append(result)
    
    # Sort by combined score
    top_queries = sorted(
        successful_queries,
        key=lambda x: x["evaluation"].get("combined_score", 0),
        reverse=True
    )[:5]  # Top 5
    
    for i, query in enumerate(top_queries):
        report.append(f"### Query {i+1} (Score: {query['evaluation'].get('combined_score', 0):.4f})\n")
        report.append(f"- Database: {query.get('db_id', '')}")
        report.append(f"- Question: {query.get('question', '')}")
        report.append(f"- Gold SQL: {query.get('gold_sql', '')}")
        report.append(f"- Generated SQL: {query.get('generated_sql', '')}")
        report.append("")
    
    # Write report to file
    report_file = output_dir / 'evaluation_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Spider dataset evaluation results')
    parser.add_argument('results_file', type=str, help='Path to the results JSON file')
    parser.add_argument('--output-dir', type=str, default='outputs/spider_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--create-plots', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--generate-report', action='store_true',
                        help='Generate detailed evaluation report')
    parser.add_argument('--compare', type=str, default=None,
                        help='Path to another results file to compare with')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Extract summary
    summary = extract_summary(results)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful Queries: {summary['successful_queries']}")
    print(f"Success Rate: {summary['success_rate'] * 100:.2f}%")
    print(f"Average Structure Score: {summary['average_structure_score']:.4f}")
    print(f"Average Execution Score: {summary['average_execution_score']:.4f}")
    print(f"Average Combined Score: {summary['average_combined_score']:.4f}")
    print(f"Total Elapsed Time: {summary['total_elapsed_time']:.2f} seconds")
    print(f"Average Time per Query: {summary['average_time_per_query']:.2f} seconds")
    
    # Create plots if requested
    if args.create_plots:
        create_plots(results, args.output_dir)
    
    # Generate report if requested
    if args.generate_report:
        generate_report(results, args.output_dir)
    
    # Compare with another results file if provided
    if args.compare:
        compare_results = load_results(args.compare)
        compare_summary = extract_summary(compare_results)
        
        print("\nComparison:")
        print(f"Metric             | Current    | Comparison | Difference")
        print(f"-------------------|------------|------------|------------")
        print(f"Total Queries      | {summary['total_queries']:10d} | {compare_summary['total_queries']:10d} | {summary['total_queries'] - compare_summary['total_queries']:+10d}")
        print(f"Successful Queries | {summary['successful_queries']:10d} | {compare_summary['successful_queries']:10d} | {summary['successful_queries'] - compare_summary['successful_queries']:+10d}")
        print(f"Success Rate       | {summary['success_rate']*100:9.2f}% | {compare_summary['success_rate']*100:9.2f}% | {(summary['success_rate'] - compare_summary['success_rate'])*100:+9.2f}%")
        print(f"Structure Score    | {summary['average_structure_score']:10.4f} | {compare_summary['average_structure_score']:10.4f} | {summary['average_structure_score'] - compare_summary['average_structure_score']:+10.4f}")
        print(f"Execution Score    | {summary['average_execution_score']:10.4f} | {compare_summary['average_execution_score']:10.4f} | {summary['average_execution_score'] - compare_summary['average_execution_score']:+10.4f}")
        print(f"Combined Score     | {summary['average_combined_score']:10.4f} | {compare_summary['average_combined_score']:10.4f} | {summary['average_combined_score'] - compare_summary['average_combined_score']:+10.4f}")
        print(f"Avg Time per Query | {summary['average_time_per_query']:10.2f} | {compare_summary['average_time_per_query']:10.2f} | {summary['average_time_per_query'] - compare_summary['average_time_per_query']:+10.2f}")

if __name__ == "__main__":
    main()