import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from collections import defaultdict
from tqdm import tqdm
import argparse
import os


class QueryEvaluator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.document_collection = {}
        self.doc_text_to_id = {}
        self.bm25 = None
        self.doc_ids = []
        self.subject_query_map = defaultdict(list)
        self.ict_pairs = []

        self.load_dataset()
        self.setup_searcher()

    def load_dataset(self):
        """Load the dataset file."""
        print("Loading dataset...")

        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            self.ict_pairs = data['ict_pairs']

        # Track relevance distribution by subject
        subject_relevance_counts = defaultdict(lambda: defaultdict(int))
        for pair in self.ict_pairs:
            subject_id = pair['metadata']['subject_id']
            relevance_score = pair['relevance_score']
            subject_relevance_counts[subject_id][relevance_score] += 1

        print("\nRelevance score distribution by subject:")
        for subject_id, counts in sorted(subject_relevance_counts.items()):
            print(f"{subject_id}: {dict(sorted(counts.items()))}")

        print("\nProcessing document collection...")
        seen_docs = {}
        for idx, pair in enumerate(self.ict_pairs):
            doc_text = pair['document_text']
            subject_id = pair['metadata']['subject_id']

            if doc_text not in seen_docs:
                doc_id = len(seen_docs)
                seen_docs[doc_text] = doc_id
                self.document_collection[doc_id] = {
                    'text': doc_text,
                    'subject_id': subject_id
                }
                self.doc_text_to_id[doc_text] = doc_id

            # Track queries by subject
            self.subject_query_map[subject_id].append(pair['query_text'])

        print(f"Processed {len(self.document_collection)} unique documents")
        print(f"Found {len(self.subject_query_map)} unique subjects")
        print(f"Total ICT pairs: {len(self.ict_pairs)}")

    def setup_searcher(self):
        """Set up BM25 searcher with the document collection."""
        print("Setting up BM25 searcher...")
        self.doc_ids = list(range(len(self.document_collection)))

        # Tokenize documents
        tokenized_corpus = []
        for doc_id in self.doc_ids:
            text = self.document_collection[doc_id]['text'].lower()
            tokens = [token.strip() for token in text.split() if token.strip()]
            tokenized_corpus.append(tokens)

        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index created with {len(tokenized_corpus)} documents")

    def calculate_ndcg(self, ranked_list: List[int], relevance_dict: Dict[int, float], k: int) -> float:
        dcg = 0
        idcg = 0

        for i, doc_id in enumerate(ranked_list[:k]):
            if doc_id in relevance_dict:
                rel = relevance_dict[doc_id]
                dcg += (2 ** rel - 1) / np.log2(i + 2)

        sorted_rels = sorted([rel for rel in relevance_dict.values()], reverse=True)
        for i, rel in enumerate(sorted_rels[:k]):
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0

    def calculate_mrr(self, ranked_list: List[int], relevance_dict: Dict[int, float], k: int) -> float:
        for i, doc_id in enumerate(ranked_list[:k]):
            if doc_id in relevance_dict and relevance_dict[doc_id] > 0:
                return 1.0 / (i + 1)
        return 0

    def evaluate_queries(self, k: int = 10, debug: bool = False) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Evaluate queries using BM25 and calculate NDCG@k and MRR@k.
        """
        results = {}
        subject_results = defaultdict(list)
        eeg_query_map = {}

        # Create mapping of text queries to their EEG data and subject
        for pair in self.ict_pairs:
            query = pair['query_text']
            subject_id = pair['metadata']['subject_id']
            if 'query_eeg' in pair and 'query_mask' in pair:
                if query not in eeg_query_map:
                    eeg_query_map[query] = {
                        'query_eeg': pair['query_eeg'],
                        'query_mask': pair['query_mask'],
                        'subject_id': subject_id
                    }

        # Group documents and their relevance scores by query and subject
        query_docs = defaultdict(lambda: defaultdict(list))
        for pair in self.ict_pairs:
            query = pair['query_text']
            subject_id = pair['metadata']['subject_id']
            doc_text = pair['document_text']
            doc_id = self.doc_text_to_id[doc_text]
            relevance = pair['relevance_score']
            query_docs[query][subject_id].append((doc_id, relevance))

        # Convert to dict with highest relevance scores
        final_query_docs = {}
        for query, subject_docs in query_docs.items():
            relevance_dict = {}
            subject_id = None
            for subj_id, doc_list in subject_docs.items():
                subject_id = subj_id  # Store subject ID for this query
                for doc_id, rel in doc_list:
                    if doc_id not in relevance_dict or rel > relevance_dict[doc_id]:
                        relevance_dict[doc_id] = rel
            final_query_docs[query] = (relevance_dict, subject_id)

        print(f"\nEvaluating queries with k={k}...")

        for query_idx, (query, (relevance_dict, subject_id)) in enumerate(tqdm(final_query_docs.items())):
            query_tokens = [token.strip() for token in query.lower().split() if token.strip()]
            doc_scores = self.bm25.get_scores(query_tokens)
            ranked_indices = np.argsort(-doc_scores)
            ranked_list = ranked_indices[:k].tolist()

            ndcg = self.calculate_ndcg(ranked_list, relevance_dict, k)
            mrr = self.calculate_mrr(ranked_list, relevance_dict, k)

            # Store results with subject information
            results[query] = {
                f'ndcg@{k}': ndcg,
                f'mrr@{k}': mrr,
                'subject_id': subject_id,
                'eeg_data': eeg_query_map.get(query, {})
            }

            # Track results by subject
            subject_results[subject_id].append({
                'query': query,
                f'ndcg@{k}': ndcg,
                f'mrr@{k}': mrr
            })

            if debug and query_idx < 2:
                print(f"\nDebug information for query: {query}")
                print(f"Subject ID: {subject_id}")
                print(f"Number of relevant documents: {sum(1 for rel in relevance_dict.values() if rel > 0)}")
                print(f"Top {k} retrieved documents:")
                for i, doc_id in enumerate(ranked_list[:k]):
                    rel_score = relevance_dict.get(doc_id, 0)
                    doc_text = self.document_collection[doc_id]['text']
                    doc_subject = self.document_collection[doc_id]['subject_id']
                    print(f"Rank {i + 1}: Doc: {doc_text[:100]}... Relevance: {rel_score}, Subject: {doc_subject}")

        return results, subject_results


def calculate_subject_metrics(subject_results: Dict[str, List[Dict[str, Any]]], k: int) -> Dict[str, Dict[str, float]]:
    """Calculate average metrics for each subject."""
    subject_metrics = {}

    for subject_id, queries in subject_results.items():
        ndcg_scores = [q[f'ndcg@{k}'] for q in queries]
        mrr_scores = [q[f'mrr@{k}'] for q in queries]

        subject_metrics[subject_id] = {
            f'mean_ndcg@{k}': np.mean(ndcg_scores),
            f'std_ndcg@{k}': np.std(ndcg_scores),
            f'mean_mrr@{k}': np.mean(mrr_scores),
            f'std_mrr@{k}': np.std(mrr_scores),
            'num_queries': len(queries)
        }

    return subject_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate search queries using BM25 ranking.')

    # Required arguments
    parser.add_argument('--input', '-i',
                        type=str,
                        required=True,
                        help='Path to the input dataset pickle file')

    # Optional arguments
    parser.add_argument('--k', '-k',
                        type=int,
                        default=10,
                        help='Number of top documents to consider (default: 10)')

    parser.add_argument('--debug', '-d',
                        action='store_true',
                        help='Enable debug output')

    parser.add_argument('--output', '-o',
                        type=str,
                        help='Output path for results (default: same as input with _evaluation_k{k} suffix)')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    try:
        # Initialize evaluator with the dataset
        evaluator = QueryEvaluator(args.input)

        # Evaluate queries
        results, subject_results = evaluator.evaluate_queries(k=args.k, debug=args.debug)

        # Calculate overall metrics
        ndcg_scores = [v[f'ndcg@{args.k}'] for v in results.values()]
        mrr_scores = [v[f'mrr@{args.k}'] for v in results.values()]
        queries_with_eeg = sum(1 for v in results.values() if v['eeg_data'])

        # Calculate subject-level metrics
        subject_metrics = calculate_subject_metrics(subject_results, args.k)

        # Print overall results
        print(f"\nOverall Evaluation Results (k={args.k}):")
        print(f"Number of queries evaluated: {len(results)}")
        print(f"Queries with EEG data: {queries_with_eeg}")
        print(f"Average NDCG@{args.k}: {np.mean(ndcg_scores):.3f} ± {np.std(ndcg_scores):.3f}")
        print(f"Average MRR@{args.k}: {np.mean(mrr_scores):.3f} ± {np.std(mrr_scores):.3f}")

        # Print subject-level results
        print("\nSubject-level Results:")
        for subject_id, metrics in subject_metrics.items():
            print(f"\nSubject {subject_id}:")
            print(f"Number of queries: {metrics['num_queries']}")
            print(f"NDCG@{args.k}: {metrics[f'mean_ndcg@{args.k}']:.3f} ± {metrics[f'std_ndcg@{args.k}']:.3f}")
            print(f"MRR@{args.k}: {metrics[f'mean_mrr@{args.k}']:.3f} ± {metrics[f'std_mrr@{args.k}']:.3f}")

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Use input path with suffix
            output_path = args.input.replace('.pkl', f'_evaluation_k{args.k}.pkl')

        # Save results
        with open(output_path, 'wb') as f:
            pickle.dump({
                'metrics': results,
                'subject_metrics': subject_metrics,
                'k': args.k,
                'summary': {
                    f'mean_ndcg@{args.k}': np.mean(ndcg_scores),
                    f'std_ndcg@{args.k}': np.std(ndcg_scores),
                    f'mean_mrr@{args.k}': np.mean(mrr_scores),
                    f'std_mrr@{args.k}': np.std(mrr_scores),
                    'total_queries': len(results),
                    'queries_with_eeg': queries_with_eeg,
                    'num_subjects': len(subject_metrics)
                }
            }, f)

        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    main()