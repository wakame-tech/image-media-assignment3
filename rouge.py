from rouge_score import rouge_scorer

sys = "I am a cat"
ref = "There is a cat on the mat"

metrics = ["rouge1", "rouge2", "rougeL"]
scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
scores = scorer.score(ref, sys)
for metric in metrics:
    print(metric, scores[metric])
