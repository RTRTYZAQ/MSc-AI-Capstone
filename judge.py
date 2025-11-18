from sacrebleu import corpus_bleu
import json

refs = [["The cat is on the mat.", "There is a cat on the mat."]]  # 可多参考
sys = ["The cat sits on the mat.", "A cat on the mat."]

with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval.json', 'r') as f:
    sys = json.load(f)

with open('/workspace/algorithm/Capstone/MSc-AI-Capstone/Dataset/eval/ground_truth.json', 'r') as f:
    refs = [_["conversations"][1]["value"]  for _ in json.load(f)]




score = corpus_bleu(sys, refs)
print("SacreBLEU:", score.score)