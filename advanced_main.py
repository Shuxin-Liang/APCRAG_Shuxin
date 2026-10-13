import json
import argparse
from typing import Dict, List
from modules.retriever import retrieve_topk_docs, set_index_paths
from modules.utils import generate_answer_from_docs
from modules.privacy_editing import IntegratedPrivacyPipeline
from modules.config import get_domain_config, get_domain_from_args, SUPPORTED_DOMAINS, print_domain_status
from modules.data_loader import load_qa_data, load_privacy_prompts, load_untargeted_attacks

def generate_privacy_aware_answer(
    query: str, 
    top_k: int = 5,
    detection_strategy: str = 'zero_shot',
    editing_strategy: str = 'generalize',
    max_iterations: int = 1,
    enable_ragas: bool = False,
    ground_truth: str = None,
    use_memory: bool = False
) -> Dict:
    """
    Generate privacy-aware answers (supports real-time RAGAS evaluation and memory functionality)
    """
    retrieved_docs = retrieve_topk_docs(query, top_k=top_k)
    
    # Use IntegratedPrivacyPipeline to process documents
    pipeline = IntegratedPrivacyPipeline(
        detection_strategy=detection_strategy,
        editing_strategy=editing_strategy,
        memory_enabled=use_memory,
        memory_db_path="memory_trajectories.json"
    )
    
    redacted_docs = []
    for doc in retrieved_docs:
        result = pipeline.process_document(query, doc, max_iterations=max_iterations)
        redacted_docs.append(result['final_passage'])
    
    answer = generate_answer_from_docs(query, redacted_docs)
    
    result = {
        'strategy': f"{detection_strategy}_{editing_strategy}",
        'answer': answer,
        'retrieved_docs': retrieved_docs,
        'redacted_docs': redacted_docs
    }
    
    # 🆕 Real-time RAGAS evaluation
    if enable_ragas and ground_truth:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
            from ragas.llms import LangchainLLMWrapper
            from langchain_openai import ChatOpenAI
            from datasets import Dataset as EvaluationDataset
            import os
            
            print(f"    🎯 Starting real-time RAGAS evaluation...")
            
            # Check API key
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                with open('openai_key.txt', 'r') as f:
                    openai_api_key = f.read().strip()
                os.environ['OPENAI_API_KEY'] = openai_api_key
            
            # Prepare RAGAS dataset
            dataset_dict = {
                'question': [query],
                'answer': [answer],
                'ground_truth': [ground_truth],
                'contexts': [retrieved_docs[:3]]  # Use first 3 documents as context
            }
            
            eval_dataset = EvaluationDataset.from_dict(dataset_dict)
            
            # Set up LLM - Adapt to RAGAS 0.1.0
            llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model="gpt-4o-mini", 
                    temperature=0,
                    openai_api_key=openai_api_key
                )
            )
            
            # RAGAS evaluation metrics - Use correct API for RAGAS 0.1.0
            ragas_metrics = [
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
            
            # Run evaluation
            ragas_result = ragas_evaluate(eval_dataset, metrics=ragas_metrics)
            
            # Add RAGAS scores to result
            result['ragas_scores'] = {}
            for metric_name, score in ragas_result.items():
                if isinstance(score, (int, float)):
                    result['ragas_scores'][metric_name] = score
                elif hasattr(score, 'mean'):
                    result['ragas_scores'][metric_name] = float(score.mean())
            
            print(f"    ✅ RAGAS evaluation completed: {result['ragas_scores']}")
            
        except Exception as e:
            print(f"    ⚠️ RAGAS evaluation failed: {e}")
            result['ragas_scores'] = {'error': str(e)}
    
    return result

def run_strategy_comparison(query: str, top_k: int = 3) -> Dict:
    """
    Run strategy comparison analysis
    """
    # 🔧 Fix: First perform retrieval to get documents for all strategies
    from modules.retriever import retrieve_topk_docs
    
    print(f"  �� Retrieving related documents...")
    retrieved_docs = retrieve_topk_docs(query, top_k=top_k)
    print(f"  ✅ Retrieved {len(retrieved_docs)} documents")
    
    # Define strategy combinations to compare
    strategy_combinations = [
        ('zero_shot', 'generalize'),
        ('static_expert', 'generalize'),
        ('few_shot', 'generalize'),
        ('cot', 'generalize')
    ]
    
    comparison_results = {
        'query': query,
        'retrieved_docs': retrieved_docs,  # 💡 Save retrieved documents
        'results': {}
    }
    
    for detection_strategy, editing_strategy in strategy_combinations:
        strategy_name = f"{detection_strategy}_{editing_strategy}"
        
        try:
            result = generate_privacy_aware_answer(
                query, 
                top_k=top_k,
                detection_strategy=detection_strategy,
                editing_strategy=editing_strategy,
                max_iterations=1,
                use_memory=False  # run_strategy_comparison does not use memory
            )
            
            comparison_results['results'][strategy_name] = result['answer']
            
        except Exception as e:
            comparison_results['results'][strategy_name] = f"Error: {str(e)}"
    
    return comparison_results

def run_strategy_comparison_with_ragas(query: str, top_k: int = 3, enable_ragas: bool = False, 
                                     ground_truth: str = None, use_memory: bool = False) -> Dict:
    """
    Run strategy comparison analysis (supports RAGAS evaluation and memory functionality)
    """
    # 🔧 Fix: First perform retrieval to get documents for all strategies
    from modules.retriever import retrieve_topk_docs
    
    print(f"  📄 Retrieving related documents...")
    retrieved_docs = retrieve_topk_docs(query, top_k=top_k)
    print(f"  ✅ Retrieved {len(retrieved_docs)} documents")
    
    # Define strategy combinations to compare
    strategy_combinations = [
        ('zero_shot', 'generalize'),
        ('static_expert', 'generalize'),
        ('few_shot', 'generalize'),
        ('cot', 'generalize')
    ]
    
    # 🆕 If memory is enabled, add memory strategy
    if use_memory:
        strategy_combinations.append(('zero_shot', 'memory_generalize'))
    
    comparison_results = {
        'query': query,
        'retrieved_docs': retrieved_docs,  # 💡 Save retrieved documents
        'results': {},
        'ragas_scores': {}  # 💡 Save RAGAS scores for all strategies
    }
    
    for detection_strategy, editing_strategy in strategy_combinations:
        strategy_name = f"{detection_strategy}_{editing_strategy}"
        
        try:
            result = generate_privacy_aware_answer(
                query, 
                top_k=top_k,
                detection_strategy=detection_strategy,
                editing_strategy=editing_strategy,
                max_iterations=1,
                enable_ragas=enable_ragas,
                ground_truth=ground_truth,
                use_memory=use_memory
            )
            
            comparison_results['results'][strategy_name] = result['answer']
            
            # If RAGAS evaluation is enabled, save scores
            if enable_ragas and 'ragas_scores' in result:
                comparison_results['ragas_scores'][strategy_name] = result['ragas_scores']
            
        except Exception as e:
            comparison_results['results'][strategy_name] = f"Error: {str(e)}"
            if enable_ragas:
                comparison_results['ragas_scores'][strategy_name] = {'error': str(e)}
    
    return comparison_results

def run_batch_queries(queries: List[str], top_k: int = 3, enable_ragas: bool = False, 
                     ground_truth_data: Dict = None, use_memory: bool = False) -> Dict:
    """
    Batch process queries (supports real-time RAGAS evaluation)
    """
    batch_results = {
        'total_queries': len(queries),
        'queries': {},
        'ragas_enabled': enable_ragas
    }
    
    for i, query in enumerate(queries, 1):
        # Ensure query is string type
        if isinstance(query, dict) and 'query' in query:
            query_str = query['query']
        elif isinstance(query, str):
            query_str = query
        else:
            print(f"⚠️ Skipping invalid query format: {query}")
            continue
        
        print(f"\n📊 Processing query {i}/{len(queries)}: {query_str[:100]}...")
        
        # Get corresponding ground truth (if any)
        ground_truth = None
        if enable_ragas and ground_truth_data:
            # Support multiple ground truth formats
            if query_str in ground_truth_data:
                ground_truth = ground_truth_data[query_str]
            elif isinstance(ground_truth_data, list) and i-1 < len(ground_truth_data):
                gt_item = ground_truth_data[i-1]
                if isinstance(gt_item, dict) and 'response' in gt_item:
                    ground_truth = gt_item['response']
                elif isinstance(gt_item, str):
                    ground_truth = gt_item
        
        comparison_results = run_strategy_comparison_with_ragas(query_str, top_k, enable_ragas, ground_truth, use_memory)
        batch_results['queries'][query_str] = {
            'answers': comparison_results['results'],
            'retrieved_docs': comparison_results['retrieved_docs'],  # 💡 Save retrieved documents
            'ragas_scores': comparison_results.get('ragas_scores', {})  # 💡 Save RAGAS scores
        }
    
    return batch_results

def main():
    parser = argparse.ArgumentParser(description='Batch privacy-aware RAG pipeline - supports three domains')
    parser.add_argument('--domain', type=str, choices=SUPPORTED_DOMAINS, 
                       help='Specify domain: finance, biometrics, legal')
    parser.add_argument('--query_type', type=str, choices=['utility', 'targeted', 'untargeted'], 
                       default='utility', help='Query type')
    parser.add_argument('--queries_file', type=str, help='Custom query file path')
    parser.add_argument('--top_k', type=int, default=3, help='Number of retrieved documents')
    parser.add_argument('--output', type=str, help='Output file name (auto-generated if not specified)')
    parser.add_argument('--faiss_index', type=str, help='FAISS index file path')
    parser.add_argument('--embeddings_file', type=str, help='Embedding vector file path')
    parser.add_argument('--chunks_file', type=str, help='Document block JSONL file path')
    parser.add_argument('--check_files', action='store_true', help='Check status of all domain files')
    parser.add_argument('--enable_ragas', action='store_true', help='Enable real-time RAGAS evaluation (only supports utility queries)')
    
    # 🆕 Memory functionality parameters
    parser.add_argument('--memory_file', type=str, help='Memory example file path (JSON format)')
    parser.add_argument('--use_memory', action='store_true', help='Enable memory functionality')
    
    args = parser.parse_args()
    
    # If only checking file status
    if args.check_files:
        print_domain_status()
        return
    
    # Determine domain
    if args.domain:
        domain = args.domain
        print(f"🎯 Using specified domain: {domain}")
    else:
        # Try to infer domain from other parameters
        domain = get_domain_from_args(
            faiss_index=args.faiss_index,
            chunks_file=args.chunks_file,
            qa_file=args.queries_file
        )
        
        if not domain:
            print("❌ Unable to determine domain, please use --domain parameter to specify")
            print(f"Supported domains: {', '.join(SUPPORTED_DOMAINS)}")
            print("Or use --check_files to check file status")
            return
        
        print(f"🔍 Automatically detected domain: {domain}")
    
    # Get domain configuration
    try:
        domain_config = get_domain_config(domain)
        print(f"📂 Using {domain} domain configuration")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Set FAISS index path
    faiss_index = args.faiss_index or domain_config['faiss_index']
    embeddings_file = args.embeddings_file or domain_config['embeddings']
    chunks_file = args.chunks_file or domain_config['chunks']
    
    set_index_paths(
        index_path=faiss_index,
        embeddings_path=embeddings_file,
        chunks_path=chunks_file
    )
    
    print(f"🔧 Configuring index paths:")
    print(f"  FAISS index: {faiss_index}")
    print(f"   Embedding vectors: {embeddings_file}")
    print(f"   Document blocks:   {chunks_file}")
    
    # Load queries based on query type
    queries = []
    output_suffix = ""
    ground_truth_data = None  # 🆕 Ground truth data for RAGAS evaluation
    
    if args.queries_file:
        # Use custom query file
        try:
            with open(args.queries_file, 'r', encoding='utf-8') as f:
                custom_data = json.load(f)
            
            if isinstance(custom_data, list):
                queries = custom_data
            elif isinstance(custom_data, dict):
                if 'queries' in custom_data:
                    queries = custom_data['queries']
                elif 'attacks' in custom_data:
                    queries = [item['query'] for item in custom_data['attacks'] if 'query' in item]
            else:
                    queries = list(custom_data.values())
            
            output_suffix = "_custom"
            print(f"📂 Loading custom query file: {len(queries)} queries")
            
        except Exception as e:
            print(f"❌ Failed to load custom query file: {e}")
            return
    
    elif args.query_type == 'utility':
        # Load utility queries (QA data)
        qa_data = load_qa_data(domain)
        if isinstance(qa_data, list):
            queries = [item['question'] for item in qa_data if 'question' in item]
            # 🆕 Prepare ground truth data for RAGAS evaluation
            ground_truth_data = {item['question']: item['response'] for item in qa_data if 'question' in item and 'response' in item}
        else:
            print(f"❌ Unable to load utility queries from QA data")
            return
        output_suffix = "_utility"
        
    elif args.query_type == 'targeted':
        # Load targeted attack queries
        queries = load_privacy_prompts(domain)
        output_suffix = "_targeted"
        
    elif args.query_type == 'untargeted':
        # Load untargeted attack queries
        queries = load_untargeted_attacks()
        output_suffix = "_untargeted"
    
    if not queries:
        print(f"❌ No {args.query_type} type queries found")
        return
    
    print(f"🔄 Starting batch processing of {len(queries)} {args.query_type} queries")
    
    # Generate output file name
    if args.output:
        output_file = args.output
    else:
        output_file = f"{domain}{output_suffix}_results.json"
    
    # 🆕 Decide whether to enable RAGAS evaluation (only supports utility queries)
    enable_ragas = (args.enable_ragas and args.query_type == 'utility' and ground_truth_data is not None)
    if enable_ragas:
        print(f"🎯 Enabling real-time RAGAS evaluation ({len(ground_truth_data)} ground truths)")
    elif args.enable_ragas and args.query_type != 'utility':
        print(f"⚠️ RAGAS evaluation only supports utility query type, current type: {args.query_type}")
    elif args.enable_ragas and ground_truth_data is None:
        print(f"⚠️ No ground truth data found, unable to enable RAGAS evaluation")
    
    # 🆕 Decide whether to enable memory functionality
    use_memory = args.use_memory or args.memory_file is not None
    if use_memory:
        import os
        # Prioritize our persistent memory system
        default_memory_file = "memory_trajectories.json"
        
        if args.memory_file:
            memory_file = args.memory_file
        elif os.path.exists(default_memory_file):
            memory_file = default_memory_file
        else:
            # Fallback to domain-specific memory file
            memory_file = f"{domain}_examples.json"
        
        if os.path.exists(memory_file):
            if memory_file == default_memory_file:
                print(f"🧠 Enabling persistent memory system: {memory_file}")
            else:
                print(f"🧠 Enabling memory functionality, using file: {memory_file}")
        else:
            print(f"⚠️ Memory file {memory_file} does not exist, using standard mode")
            use_memory = False
    
    # Batch process queries
    batch_results = run_batch_queries(queries, args.top_k, enable_ragas, ground_truth_data, use_memory)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Batch results saved to {output_file}")
    print(f"📊 Processed a total of {batch_results['total_queries']} queries")
    
    # 🆕 If RAGAS evaluation is enabled, display evaluation result summary
    if enable_ragas and batch_results.get('ragas_enabled'):
        print(f"\n🎯 RAGAS evaluation summary:")
        ragas_summary = {}
        for query, result in batch_results['queries'].items():
            for strategy, scores in result.get('ragas_scores', {}).items():
                if strategy not in ragas_summary:
                    ragas_summary[strategy] = []
                if isinstance(scores, dict) and 'error' not in scores:
                    ragas_summary[strategy].append(scores)
        
        for strategy, scores_list in ragas_summary.items():
            if scores_list:
                avg_scores = {}
                for score_dict in scores_list:
                    for metric, value in score_dict.items():
                        if metric not in avg_scores:
                            avg_scores[metric] = []
                        avg_scores[metric].append(value)
                
                print(f"  📈 {strategy}:")
                for metric, values in avg_scores.items():
                    avg_value = sum(values) / len(values)
                    print(f"    {metric}: {avg_value:.4f}")
    
    print(f"\n💡 Use the following command for evaluation:")
    
    if args.query_type == 'utility':
        print(f"   python evaluate_separated.py --domain {domain} --results_file {output_file}")
    elif args.query_type in ['targeted', 'untargeted']:
        print(f"   python evaluate_separated.py --domain {domain} --privacy_results_file {output_file}")

if __name__ == "__main__":
    main() 