import argparse
import pandas as pd
import os
from trectools import TrecQrel, TrecRun, TrecEval


def main():
    parser = argparse.ArgumentParser(description='Search in the indexed data')
    parser.add_argument('-i', '--input',
                        default='./search_results/bge-base_wikitables.csv',
                        help='Queries file')
    parser.add_argument('-o', '--output', default='./evaluate_results/',
                        help='Output folder')
    args = parser.parse_args()

    qrels_file = "./benchmarks_wikitables/qrels.txt"
    qrels = TrecQrel(qrels_file)

    # Use trec_eval to calculate the metrics
    search_results = TrecRun(args.input)
    te = TrecEval(search_results, qrels)

    p1 = te.get_precision(depth=1, per_query=True)
    p5 = te.get_precision(depth=5, per_query=True)
    p10 = te.get_precision(depth=10, per_query=True)
    map = te.get_map(per_query=True)
    ndcg5 = te.get_ndcg(depth=5, per_query=True)
    ndcg10 = te.get_ndcg(depth=10, per_query=True)

    queries = pd.read_csv('./benchmarks_wikitables/queries.txt',
                          sep='	', header=None)
    q_rels = pd.read_csv('./benchmarks_wikitables/qrels.txt',
                         sep='	', header=None)
    rank = pd.read_csv(args.input, sep='	')

    #  Calculate MRR
    all_rr = []
    q_ids = []
    for _, (q_id, _) in queries.iterrows():
        r_rank = rank.loc[rank.iloc[:, 0] == q_id]
        rr = 0

        for doc in r_rank.iloc[:, 2].tolist():
            if doc in q_rels.iloc[:, 2].tolist():
                pos = r_rank.loc[rank.iloc[:, 2] == doc].iloc[:, 3]
                rr = pos.tolist()[0]/20
                break
        q_ids.append(str(q_id))
        all_rr.append(rr)

    mrr = pd.DataFrame(all_rr, index=q_ids, columns=["MRR"])
    mrr.index.name = 'query'

    # Group all results and save into CSV
    final = pd.concat([p1, p5, p10, map, ndcg5, ndcg10], axis=1)
    final = pd.merge(final, mrr, left_index=True, right_index=True)
    final = final.fillna(0)
    final.loc['mean'] = final.mean()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    final.to_csv(args.output + args.input.split('/')[-1])


if __name__ == "__main__":
    main()
