from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str)
parser.add_argument('-t', type=str)
parser.add_argument('-o', type=str)
parser.add_argument('--num_samples', type=int)
parser.add_argument("--qe_vec_path", type=str)
args = parser.parse_args()

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

with open(args.s, "r") as f:
    srcs = f.readlines()
    srcs = [src for src in srcs for _ in range(args.num_samples)]
with open(args.t, "r") as f:
    hypos = f.readlines()
data = [{"src": src, "mt": hypo} for src, hypo in zip(srcs, hypos)]


qe_result = []

model_output = model.predict(data, batch_size=128, gpus=1)
qe_scores = np.array(model_output.scores)
qe_scores = qe_scores.reshape((len(hypos)//args.num_samples, args.num_samples))

hypos = [hypos[i: i+args.num_samples] for i in range(0, len(hypos), args.num_samples)]
for i, hypo in enumerate(hypos):
    candidate_id = qe_scores[i].argmax()
    qe_result.append(hypo[candidate_id])

with open(args.o, "w") as f:
    f.writelines(qe_result)
np.save(args.qe_vec_path, qe_scores)
print("qe score done", qe_scores.shape)
