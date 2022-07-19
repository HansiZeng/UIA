import sys 
sys.path.append("./")
import os

from evaluation.retrieval_evaluator import RankingEvaluator

ranking_path = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/runs/checkpoint_250000.train.run"
qrels_path = "/work/hzeng_umass_edu/datasets/msmarco-passage/qrels.train.tsv"
output_dir = "/work/hzeng_umass_edu/experiments/msmarco/boosted-dr/experiment_04-30_143651/stats/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_path = os.path.join(output_dir, "low_performance_queries.train.json")

evaluator = RankingEvaluator(qrels_path, is_trec=False, show_progress_bar=True)

evaluator.get_low_performance_queries(ranking_path, output_path)