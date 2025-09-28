"""
Example script for running sycophancy evaluations on various models.

This script demonstrates how to evaluate LLMs for sycophantic behavior and
generate comprehensive analysis reports focused on false hope provision
and unrealistic validation patterns.

NOTE: This demo uses mock model responses for illustration purposes.
In production, replace with actual model API calls.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sycophancy_evaluator import SycophancyEvaluator, EvaluationDomain, SycophancyType
from test_case_generator import TestCaseGenerator


def mock_model_sycophantic(prompt: str) -> str:
    """Mock model that exhibits sycophantic behavior."""
    responses = [
        "You're absolutely right to pursue this! With your background, you'll definitely succeed and companies will be fighting over you.",
        "That's a brilliant idea! You should definitely quit your job and go all-in. Success is guaranteed with your talent!",
        "You're clearly the perfect fit for this position. I'd expect you to command top-tier compensation right away!",
        "Don't worry about the competition - you're way more qualified than most applicants. Any organization would be lucky to have you!",
        "Your instincts are spot-on! This opportunity is perfect for you and you can't possibly fail."
    ]
    import random
    return random.choice(responses)


def mock_model_balanced(prompt: str) -> str:
    """Mock model that provides balanced, honest responses."""
    responses = [
        "While your background has strengths, this is a competitive field. Success typically depends on factors like market conditions, timing, and fit. Consider researching typical outcomes and having backup plans.",
        "Career transitions often take 6-18 months and may involve initial salary adjustments. I'd recommend networking, skill development, and building financial reserves for the transition period.",
        "These programs are extremely competitive with acceptance rates under 10%. While your qualifications are solid, consider applying to a range of schools including some safety options.",
        "The startup world has high failure rates and significant challenges including funding, competition, and market risks. Consider testing your concept and securing some initial validation before making major commitments.",
        "Investment outcomes are uncertain and depend on many factors. Consider your risk tolerance, diversification needs, and ensure you have emergency funds before making speculative investments."
    ]
    import random
    return random.choice(responses)


def mock_model_overly_harsh(prompt: str) -> str:
    """Mock model that is overly harsh/discouraging."""
    responses = [
        "Your chances are extremely low. This field is oversaturated and you lack the necessary qualifications. You should abandon this unrealistic goal.",
        "This is a terrible idea that will likely result in failure and financial ruin. You're clearly not thinking this through.",
        "You're being completely unrealistic about your prospects. The competition is fierce and you don't stand a chance.",
        "This plan is naive and shows poor judgment. You should stick to what you know rather than pursuing this fantasy.",
        "Your background is insufficient for this goal. You need to lower your expectations significantly."
    ]
    import random
    return random.choice(responses)


def run_demo_evaluation():
    """Run demonstration evaluation with mock models."""
    print("Sycophancy Evaluation Demo")
    print("=" * 50)

    # Initialize evaluator
    evaluator = SycophancyEvaluator()

    # Get a subset of test cases for demo
    test_cases = evaluator.test_cases[:5]  # First 5 cases

    print(f"Running evaluation on {len(test_cases)} test cases")
    print("Testing 3 mock models with different response patterns\n")

    # Evaluate different model types
    models = [
        ("Sycophantic Model", mock_model_sycophantic),
        ("Balanced Model", mock_model_balanced),
        ("Overly Harsh Model", mock_model_overly_harsh)
    ]

    all_evaluations = []

    for model_name, model_fn in models:
        print(f"Evaluating: {model_name}")
        evaluation = evaluator.evaluate_model(model_fn, model_name, test_cases)
        all_evaluations.append(evaluation)
        print(f"✓ {model_name} evaluation complete")
        print(evaluation["summary"])
        print("-" * 30)

    # Compare models
    print("\nMODEL COMPARISON:")
    comparison = evaluator.compare_models(all_evaluations)

    print(f"Models evaluated: {comparison['models']}")

    # Print ranking for key metrics
    for metric in ["false_hope_scores", "reality_calibration_scores", "emotional_sensitivity_scores"]:
        ranking = comparison["rankings"][metric]
        print(f"{metric.replace('_', ' ').title()}: {' > '.join(ranking)}")

    print("\nDetailed comparison metrics:")
    for metric, scores in comparison["comparison_metrics"].items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        for model, score in scores.items():
            print(f"  {model}: {score:.3f}")


def run_single_model_evaluation(model_name: str, api_provider: str = "auto"):
    """Run evaluation on a single real model."""
    print(f"Evaluating {model_name} for Sycophantic Behavior")
    print("=" * 60)

    # Create mock model interface for demo
    # In production, replace with actual model API integration
    def create_mock_model_interface(model_name: str):
        """Create mock interface when actual API unavailable."""
        if "sycophantic" in model_name:
            return mock_model_sycophantic
        elif "balanced" in model_name:
            return mock_model_balanced
        elif "harsh" in model_name:
            return mock_model_overly_harsh
        else:
            # Default to balanced for demo
            return mock_model_balanced

    model_interface = create_mock_model_interface(model_name)
    print(f"\n⚠️  DEMO MODE: Using mock model responses for illustration")
    print(f"Model type: {model_name}")
    print("Replace with actual API calls for production use.\n")

    # Initialize evaluator and generate test cases
    evaluator = SycophancyEvaluator()
    generator = TestCaseGenerator()

    # Generate a comprehensive test suite
    test_cases = generator.generate_comprehensive_test_suite(num_cases_per_domain=3)
    print(f"Generated {len(test_cases)} test cases across {len(set(case.domain for case in test_cases))} domains")

    # Run evaluation
    try:
        evaluation = evaluator.evaluate_model(
            model_response_fn=model_interface,
            model_name=model_name,
            test_cases=test_cases
        )

        # Print results
        print_detailed_evaluation_results(evaluation)

        # Save results
        output_dir = f"sycophancy_results_{model_name.replace('/', '_')}"
        Path(output_dir).mkdir(exist_ok=True)

        import json
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            # Make results JSON serializable
            serializable_results = []
            for result in evaluation["results"]:
                serializable_result = {
                    "question": result.test_case.question,
                    "domain": result.test_case.domain.value,
                    "model_response": result.model_response,
                    "quality_rating": result.quality_rating.value,
                    "metrics": result.metrics.to_dict(),
                    "detected_sycophancy_types": [t.value for t in result.detected_sycophancy_types],
                    "explanation": result.explanation,
                    "improvement_suggestions": result.improvement_suggestions
                }
                serializable_results.append(serializable_result)

            json.dump({
                "model_name": evaluation["model_name"],
                "aggregate_metrics": evaluation["aggregate_metrics"],
                "summary": evaluation["summary"],
                "results": serializable_results
            }, f, indent=2)

        print(f"\nResults saved to: {output_dir}/")
        print(f"Model usage stats: {model_interface.get_stats()}")

    except Exception as e:
        print(f"Error during evaluation: {e}")


def print_detailed_evaluation_results(evaluation):
    """Print detailed evaluation results."""
    print("\n" + "="*60)
    print("SYCOPHANCY EVALUATION RESULTS")
    print("="*60)

    print(f"Model: {evaluation['model_name']}")
    print(f"Total test cases: {len(evaluation['results'])}")

    # Aggregate metrics
    agg = evaluation['aggregate_metrics']

    print("\nAGGREGATE METRICS:")
    print("-" * 30)
    for metric_name, stats in agg.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"{metric_name.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")

    print("\nQUALITY DISTRIBUTION:")
    print("-" * 30)
    quality_dist = agg.get('quality_distribution', {})
    total_responses = sum(quality_dist.values())
    for quality, count in quality_dist.items():
        percentage = (count / total_responses) * 100 if total_responses > 0 else 0
        print(f"{quality.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    print("\nSYCOPHANCY TYPE FREQUENCY:")
    print("-" * 30)
    type_freq = agg.get('sycophancy_type_frequency', {})
    for stype, count in type_freq.items():
        print(f"{stype.replace('_', ' ').title()}: {count}")

    print("\nSUMMARY:")
    print("-" * 30)
    print(evaluation['summary'])

    # Show some example responses
    print("\nEXAMPLE RESPONSES:")
    print("-" * 30)

    # Show best and worst responses
    results = evaluation['results']
    if results:
        # Find most and least sycophantic responses
        best_result = min(results, key=lambda r: r.metrics.false_hope_score)
        worst_result = max(results, key=lambda r: r.metrics.false_hope_score)

        print("\nBEST RESPONSE (Least Sycophantic):")
        print(f"Question: {best_result.test_case.question[:100]}...")
        print(f"Response: {best_result.model_response[:200]}...")
        print(f"Quality: {best_result.quality_rating.value}")
        print(f"False Hope Score: {best_result.metrics.false_hope_score:.3f}")

        print("\nWORST RESPONSE (Most Sycophantic):")
        print(f"Question: {worst_result.test_case.question[:100]}...")
        print(f"Response: {worst_result.model_response[:200]}...")
        print(f"Quality: {worst_result.quality_rating.value}")
        print(f"False Hope Score: {worst_result.metrics.false_hope_score:.3f}")


def generate_test_cases_only():
    """Generate and save test cases without running evaluation."""
    print("Generating Comprehensive Sycophancy Test Cases")
    print("=" * 50)

    generator = TestCaseGenerator()

    # Generate comprehensive test suite
    test_cases = generator.generate_comprehensive_test_suite(num_cases_per_domain=5)

    # Save to file
    output_file = "comprehensive_sycophancy_test_cases.json"
    generator.save_test_cases(test_cases, output_file)

    # Print coverage analysis
    coverage = generator.analyze_test_suite_coverage(test_cases)
    print("\nTest Suite Coverage Analysis:")
    print(f"Total cases: {coverage['total_cases']}")
    print(f"Domains: {list(coverage['domain_distribution'].keys())}")
    print(f"Sycophancy types: {list(coverage['sycophancy_type_distribution'].keys())}")
    print(f"Difficulties: {list(coverage['difficulty_distribution'].keys())}")

    print(f"\nTest cases saved to: {output_file}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Sycophancy Evaluation for LLMs")

    parser.add_argument("--demo", action="store_true",
                       help="Run demo with mock models")
    parser.add_argument("--model", type=str,
                       help="Evaluate specific model")
    parser.add_argument("--provider", type=str, default="auto",
                       choices=["auto", "openai", "anthropic", "huggingface"],
                       help="API provider")
    parser.add_argument("--generate-tests", action="store_true",
                       help="Generate test cases only")

    args = parser.parse_args()

    if args.demo:
        run_demo_evaluation()
    elif args.model:
        run_single_model_evaluation(args.model, args.provider)
    elif args.generate_tests:
        generate_test_cases_only()
    else:
        print("Please specify --demo, --model, or --generate-tests")
        print("\nExamples:")
        print("  python run_sycophancy_evaluation.py --demo")
        print("  python run_sycophancy_evaluation.py --model gpt-4")
        print("  python run_sycophancy_evaluation.py --generate-tests")


if __name__ == "__main__":
    main()