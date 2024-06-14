import torch
import codecs
import json
import pandas as pd

from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline
from tqdm import tqdm


class dataset(Dataset):
    def __init__(self, examples):
        super(dataset, self).__init__()
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def collate_fn(examples):
    ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    position_sep = torch.tensor(position_sep, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels

def collate_fn_adv(examples):
    ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels = map(list, zip(*examples))

    ids_sent1 = torch.tensor(ids_sent1, dtype=torch.long)
    segs_sent1 = torch.tensor(segs_sent1, dtype=torch.long)
    att_mask_sent1 = torch.tensor(att_mask_sent1, dtype=torch.long)
    position_sep = torch.tensor(position_sep, dtype=torch.long)

    return ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels


class DataProcessor:

  def __init__(self,config):
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
    self.max_sent_len = config["max_sent_len"]

  def __str__(self,):
    pattern = """General data processor: \n\n Tokenizer: {}\n\nMax sentence length: {}""".format(self.config["model_name"], self.max_sent_len)
    return pattern

  def _get_examples(self, dataset, dataset_type="train"):
    examples = []

    for row in tqdm(dataset, desc="tokenizing..."):
      id, sentence1, sentence2, label = row

      """
      for the first sentence
      """

      sentence1_length = len(self.tokenizer.encode(sentence1))
      sentence2_length = len(self.tokenizer.encode(sentence2))

      ids_sent1 = self.tokenizer.encode(sentence1, sentence2)
      segs_sent1 = [0] * sentence1_length + [1] * (sentence2_length)
      position_sep = [1] * len(ids_sent1)
      position_sep[sentence1_length] = 1
      position_sep[0] = 0
      position_sep[1] = 1

      assert len(ids_sent1) == len(position_sep)
      assert len(ids_sent1) == len(segs_sent1)

      pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]

      if len(ids_sent1) < self.max_sent_len:
        res = self.max_sent_len - len(ids_sent1)
        att_mask_sent1 = [1] * len(ids_sent1) + [0] * res
        ids_sent1 += [pad_id] * res
        segs_sent1 += [0] * res
        position_sep += [0] * res
      else:
        ids_sent1 = ids_sent1[:self.max_sent_len]
        segs_sent1 = segs_sent1[:self.max_sent_len]
        att_mask_sent1 = [1] * self.max_sent_len
        position_sep = position_sep[:self.max_sent_len]

      example = [ids_sent1, segs_sent1, att_mask_sent1, position_sep, label]

      examples.append(example)

    print(f"finished preprocessing examples in {dataset_type}")

    return examples

class DiscourseMarkerProcessor(DataProcessor):

  def __init__(self, config):
    super(DiscourseMarkerProcessor, self).__init__(config)

    self.mapping = self.load_json('json/word_target.json')
    self.id_to_word = self.load_json('json/id_to_word.json')

  def load_json(self, path):
    try:
        with open(path, 'r') as file:
            mapping = json.load(file)
    except:
       raise FileNotFoundError(f"File {path} not found")

    return mapping

  def process_dataset(self, dataset, name="train"):
    result = []
    new_dataset = []

    for sample in dataset:
      if self.id_to_word[sample["label"]] not in self.mapping.keys():
        continue

      new_dataset.append([sample["sentence1"], sample["sentence2"], self.mapping[self.id_to_word[sample["label"]]]])

    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    labels = []

    for i, sample in tqdm(enumerate(new_dataset), desc="processing labels..."):
      labels.append([sample[-1]])

    print("one hot encoding...")
    labels = one_hot_encoder.fit_transform(labels)

    for i, (sample, label) in tqdm(enumerate(zip(new_dataset, labels)), desc="creating results..."):
      result.append([f"{name}_{i}", sample[0], sample[1], label])

    examples = self._get_examples(result, name)
    return examples


class StudentEssayProcessor(DataProcessor):

  def __init__(self, config):
    super(StudentEssayProcessor,self).__init__(config)


  def read_input_files(self, file_path, name="train", pipe=None):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      sentences = []
      label_distribution=[]
      target = []

      target_sentences = []

      id=[]

      with codecs.open(file_path, encoding="ISO-8859-1", mode="r") as f:
        for line in f:
              line = line.replace("\n","")
              line = line.split("\t")

              if line == ['\r']:
                    continue
              sample_id = line[0]
              sent = line[1].strip()
              target = line[3].strip()

              if pipe is not None:
                ds_marker = pipe(f"{sent}</s></s>{target}")[0]["label"]
                ds_marker = ds_marker.replace("_", " ")
                ds_marker = ds_marker[0].upper() + ds_marker[1:]
                target = target[0].lower() + target[1:]
                target = ds_marker + " " + target

              label = line[-1].strip()

              sentences.append(sent)
              target_sentences.append(target)
              id.append(sample_id)

              l=[0,0]
              if label == 'supports':
                    l=[1,0]
              elif label == 'attacks':
                    l=[0,1]
              label_distribution.append(l)

      result = []
      for i in range(len(label_distribution)):
        result.append([id[i],sentences[i],target_sentences[i], label_distribution[i]])

      examples = self._get_examples(result, name)

      return examples


class DebateProcessor(DataProcessor):

  def __init__(self, config):
    super(DebateProcessor,self).__init__(config)

  def read_input_files(self, file_path, name="train", pipe=None):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """
      sentences = []
      label_distribution=[]
      target_sentences = []

      id=[]

      with codecs.open(file_path, encoding="ISO-8859-1", mode="r") as f:
        for line in f:

              line = line.replace("\n","")
              line = line.split("\t")

              if line == ['\r']:
                      continue

              sample_id = line[0]
              sent = line[1].strip()
              target = line[3].strip()

              label = line[-1].strip()

              if pipe is not None:
                ds_marker = self.pipe(f"{sent}</s></s>{target}")[0]["label"]
                ds_marker = ds_marker.replace("_", " ")
                ds_marker = ds_marker[0].upper() + ds_marker[1:]
                target = target[0].lower() + target[1:]
                target = ds_marker + " " + target

              sentences.append(sent)
              target_sentences.append(target)
              id.append(sample_id)

              l=[0,0]
              if label == 'support':
                    l=[1,0]
              elif label == 'attack':
                    l=[0,1]
              label_distribution.append(l)

      result = []
      for i in range(len(label_distribution)):
        result.append([id[i],sentences[i],target_sentences[i], label_distribution[i]])

      examples = self._get_examples(result, name)

      return examples


class MARGProcessor(DataProcessor):

  def __init__(self, config):
    super(MARGProcessor, self).__init__(config)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

  def read_input_files(self, file_path, name="train", pipe=None):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      sentences = []
      label_distribution=[]
      target_sentences = []

      id=[]

      df = pd.read_csv(file_path)
      for i,row in df.iterrows():
              if row[-1] != name:
                continue

              sample_id = row[0]
              sent = row[1].strip()
              target = row[2].strip()

              if pipe is not None:
                ds_marker = self.pipe(f"{sent}</s></s>{target}")[0]["label"]
                ds_marker = ds_marker.replace("_", " ")
                ds_marker = ds_marker[0].upper() + ds_marker[1:]
                target = target[0].lower() + target[1:]
                target = ds_marker + " " + target

              label = row[3].strip()

              sentences.append(sent)
              target_sentences.append(target)
              id.append(sample_id)

              l=[0,0,0]
              if label == 'support':
                l = [1,0,0]
              elif label == 'attack':
                l = [0,1,0]
              elif label == 'neither':
                l = [0,0,1]

              label_distribution.append(l)

      result = []
      for i in range(len(label_distribution)):
        result.append([id[i],sentences[i],target_sentences[i], label_distribution[i]])

      examples = self._get_examples(result, name)

      return examples


class StudentEssayWithDiscourseInjectionProcessor(StudentEssayProcessor):

  def __init__(self, config):
    super(StudentEssayWithDiscourseInjectionProcessor, self).__init__(config)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

  def read_input_files(self, file_path, name="train"):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      examples = super().read_input_files(file_path, name, pipe=self.pipe)

      return examples


class DebateWithDiscourseInjectionProcessor(DebateProcessor):

  def __init__(self, config):
    super(DebateWithDiscourseInjectionProcessor,self).__init__(config)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")


  def read_input_files(self, file_path, name="train"):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      examples = super().read_input_files(file_path, name, pipe=self.pipe)

      return examples


class MARGWithDiscourseInjectionProcessor(DataProcessor):

  def __init__(self, config):
    super(MARGWithDiscourseInjectionProcessor,self).__init__(config)
    self.pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

  def read_input_files(self, file_path, name="train"):
      """
      Reads input files in tab-separated format.
      Will split file_paths on comma, reading from multiple files.
      """

      examples = super().read_input_files(file_path, name, pipe=self.pipe)

      return examples
