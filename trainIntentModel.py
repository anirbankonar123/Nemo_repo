import json
import os
import nemo

from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager
from nemo.utils import logging
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
import torch
import wget

#pretrained_model = nemo_nlp.models.ZeroShotIntentModel.from_pretrained("zeroshotintent_en_bert_base_uncased")

# queries = [
#     "What is the weather in Santa Clara tomorrow morning?",
#     "I'd like a veggie burger and fries",
#     "Bring me some ice cream when it's sunny"
#
# ]
#
# candidate_labels = ['Food order', 'Weather query', "Play music"]
#
# predictions = pretrained_model.predict(queries, candidate_labels, batch_size=4, multi_label=True)
#
# print('The prediction results of some sample queries with the trained model:')
# for query in predictions:
#     print(json.dumps(query, indent=4))
#
# predictions = pretrained_model.predict(queries, candidate_labels, batch_size=4, multi_label=False)
#
# print('The prediction results of some sample queries with the trained model:')
# for query in predictions:
#     print(json.dumps(query, indent=4))
#
# predictions = pretrained_model.predict(queries, candidate_labels, batch_size=4, multi_label=False,
#                                       hypothesis_template="a person is asking something related to {}")
#
# print('The prediction results of some sample queries with the trained model:')
# for query in predictions:
#     print(json.dumps(query, indent=4))
DATA_DIR = '/home/anish/Downloads'
num_examples = 5
df = pd.read_csv(os.path.join(DATA_DIR, "MNLI", "dev_matched.tsv"), sep="\t")[:num_examples]
for sent1, sent2, label in zip(df['sentence1'].tolist(), df['sentence2'].tolist(), df['gold_label'].tolist()):
    print("sentence 1: ", sent1)
    print("sentence 2: ", sent2)
    print("label: ", label)
    print("===================")

BRANCH = 'main'
WORK_DIR = "."  # you can replace WORK_DIR with your own location
#wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/zero_shot_intent_recognition/conf/zero_shot_intent_config.yaml', WORK_DIR)

# print content of the config file
config_file = os.path.join(WORK_DIR, "zero_shot_intent_config.yaml")
config = OmegaConf.load(config_file)
print(OmegaConf.to_yaml(config))

OUTPUT_DIR = "nemo_output"
config.exp_manager.exp_dir = OUTPUT_DIR
config.model.dataset.data_dir = os.path.join(DATA_DIR, "MNLI")
config.model.train_ds.file_name = "train.tsv"
config.model.validation_ds.file_path = "dev_matched.tsv"

print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

# lets modify some trainer configs
# checks if we have GPU available and uses it
print("CUDA AV:")
print(torch.cuda.is_available())
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1
config.trainer.accelerator = accelerator

config.trainer.precision = 16 if torch.cuda.is_available() else 32

# for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
# config.trainer.amp_level = O1

# remove distributed training flags
config.trainer.strategy = 'auto'

# setup max number of steps to reduce training time for demonstration purposes of this tutorial
config.trainer.max_steps = 128

trainer = pl.Trainer(**config.trainer)

exp_dir = exp_manager(trainer, config.get("exp_manager", None))

# the exp_dir provides a path to the current experiment for easy access
exp_dir = str(exp_dir)
print(exp_dir)

# get the list of supported BERT-like models, for the complete list of HugginFace models, see https://huggingface.co/models
print(nemo_nlp.modules.get_pretrained_lm_models_list(include_external=True))

# specify BERT-like model, you want to use, for example, "megatron-bert-345m-uncased" or 'bert-base-uncased'
PRETRAINED_BERT_MODEL = "albert-base-v1"
# add the specified above model parameters to the config
config.model.language_model.pretrained_model_name = PRETRAINED_BERT_MODEL

model = nemo_nlp.models.ZeroShotIntentModel(cfg=config.model, trainer=trainer)

trainer.fit(model)

# reload the saved model
saved_model = os.path.join(exp_dir, "checkpoints/ZeroShotIntentRecognition.nemo")
eval_model = nemo_nlp.models.ZeroShotIntentModel.restore_from(saved_model)

queries = [
    "I'd like a veggie burger and fries",
    "Turn off the lights in the living room",
]

candidate_labels = ['Food order', 'Play music', 'Request for directions', 'Change lighting', 'Calendar query']

predictions = eval_model.predict(queries, candidate_labels, batch_size=4, multi_label=True)

print('The prediction results of some sample queries with the trained model:')
for query in predictions:
    print(json.dumps(query, indent=4))
print("Inference finished!")