import torch
import datasets
import pickle

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import AdamW

from utils import set_random_seeds, get_config, get_device
from data_processor import StudentEssayProcessor, StudentEssayWithDiscourseInjectionProcessor,\
                            DebateProcessor, DebateWithDiscourseInjectionProcessor,\
                            MARGProcessor, MARGWithDiscourseInjectionProcessor,\
                            DiscourseMarkerProcessor, dataset,\
                            collate_fn, collate_fn_adv
from batch_sampler import BalancedSampler
from models import AdversarialNet, BaselineModel
from train import Trainer


def run():
  config = get_config()
  device = get_device()
  set_random_seeds(config["seed"])

  if config["dataset"] == "student_essay":
    if config["injection"]:
      processor = StudentEssayWithDiscourseInjectionProcessor(config)
    else:
      processor = StudentEssayProcessor(config)

    path_train = "./data/student_essay/train_essay.txt"
    path_dev = "./data/student_essay/dev_essay.txt"
    path_test = "./data/student_essay/test_essay.txt"
  elif config["dataset"] == "debate":
    if config["injection"]:
      processor = DebateWithDiscourseInjectionProcessor(config)
    else:
      processor = DebateProcessor(config)

    path_train = "./data/debate/train_debate_concept.txt"
    path_dev = "./data/debate/dev_debate_concept.txt"
    path_test = "./data/debate/test_debate_concept.txt"
  elif config["dataset"] == "m-arg":
    if config["injection"]:
      processor = MARGWithDiscourseInjectionProcessor(config)
    else:
      processor = MARGProcessor(config)

    path_train = "./data/m-arg/presidential_final.csv"
    path_dev = path_train
    path_test = path_train
  else:
    raise ValueError(f"{config['dataset']} is not a valid database name (choose between 'student_essay', 'debate' and 'm-arg')")

  data_train = processor.read_input_files(path_train, name="train")
  if config["dataset"] == "nk":
    data_dev = data_train[:len(data_train) // 10]
    data_test = data_train[-(len(data_train) // 10):]
    data_train = data_train[(len(data_train) // 10) : -(len(data_train) // 10)]
  else:
    data_dev = processor.read_input_files(path_dev, name="dev")
    data_test = processor.read_input_files(path_test, name="test")

  if config["adversarial"] or config["discovery_finetuning"]:
    df = datasets.load_dataset("discovery","discovery", trust_remote_code=True)
    adv_processor = DiscourseMarkerProcessor(config)
    if not config["dataset_from_saved"]:
      print("processing discourse marker dataset...")
      train_adv = adv_processor.process_dataset(df["train"])
      with open("./adv_dataset.pkl", "wb") as writer:
        pickle.dump(train_adv, writer)
    else:
      with open("./adv_dataset.pkl", "rb") as reader:
        train_adv = pickle.load(reader)

    data_train_tot = data_train + train_adv
  else:
    data_train_tot = data_train

  train_set = dataset(data_train_tot)
  dev_set = dataset(data_dev)
  test_set = dataset(data_test)

  if not config["adversarial"]:
    train_dataloader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    model = BaselineModel(config)
  else:
    sampler_train = BalancedSampler(data_train, train_adv, config["batch_size"])
    train_dataloader = DataLoader(train_set, batch_sampler=sampler_train, collate_fn=collate_fn_adv)

    model = AdversarialNet(config)

    if config["visualize"]:
        try:
            model.load_state_dict(torch.load(f"./{config['dataset']}_model.pt"))
            model.eval()
        except:
          raise FileNotFoundError(f"Model \"./{config['dataset']}_model.pt\" does not exist. Train the model first, then you can visualize the embeddings")

  model.to(device)

  dev_dataloader = DataLoader(dev_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_concatenated)
  test_dataloader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_concatenated)

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": 0.01,
    },
    {
      "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      "weight_decay": 0.0
    },
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"], weight_decay=config["weight_decay"])

  class_weight = config[config["dataset"]]["class_weight"]
  loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to(device))

  best_dev_f1 = -1
  result_metrics = []

  trainer = Trainer(config, device)

  if config["visualize"]:
    trainer.visualize(model, test_dataloader)
  elif config["grid_search"]:
    range_disc = np.arange(0,1.2,0.2)
    range_adv = np.arange(0,1.2,0.2)

    for discovery_weight in range_disc:
      for adv_weight in range_adv:
        for epoch in range(config["epochs"]):
          print('===== Start training: epoch {} ====='.format(epoch + 1))
          print(f"*** trying with discovery_weight = {discovery_weight}, adv_weight = {adv_weight}")
          trainer.train(epoch, model, loss_fn, optimizer, train_dataloader, discovery_weight=discovery_weight, adv_weight=adv_weight)
          dev_a, dev_p, dev_r, dev_f1 = trainer.val(model, dev_dataloader)
          test_a, test_p, test_r, test_f1 = trainer.val(model, test_dataloader)
          if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_acc, best_test_pre, best_test_rec, best_test_f1 = test_a, test_p, test_r, test_f1
            torch.save(model.state_dict(), f"./{config['dataset']}_model.pt")

        print('*** best result on test set ***')
        print(best_test_acc)
        print(best_test_pre)
        print(best_test_rec)
        print(best_test_f1, end="\n")

        result_metrics.append([best_test_acc, best_test_pre, best_test_rec, best_test_f1])

        #we reset the model and optimizer in order to start from the same random seed
        #this makes the results reproducible even without running gridsearch

        del model
        del optimizer

        set_random_seeds(config["seed"])
        model = AdversarialNet()
        model = model.to(device)

        optimizer_grouped_parameters = [
          {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
          },
          {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
          },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"], weight_decay=config["weight_decay"])

        best_dev_f1 = -1
  else:
    for epoch in range(config["epochs"]):
      print('===== Start training: epoch {} ====='.format(epoch + 1))
      trainer.train(epoch, model, loss_fn, optimizer, train_dataloader, discovery_weight=config["discovery_weight"], adv_weight=config["adv_weight"])
      dev_a, dev_p, dev_r, dev_f1 = trainer.val(model, dev_dataloader)
      test_a, test_p, test_r, test_f1 = trainer.val(model, test_dataloader)
      if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_test_acc, best_test_pre, best_test_rec, best_test_f1 = test_a, test_p, test_r, test_f1
        torch.save(model.state_dict(), f"./{config['dataset']}_model.pt")

    print('*** best result on test set ***')
    print(best_test_acc)
    print(best_test_pre)
    print(best_test_rec)
    print(best_test_f1, end="\n")
    result_metrics.append([best_test_acc, best_test_pre, best_test_rec, best_test_f1])

  print(f"Overall metrics: {result_metrics}")

if __name__ == "__main__":
  run()
