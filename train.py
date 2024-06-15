import torch
import time

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import output_metrics

class Trainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def train(self, epoch, model, loss_fn, optimizer, train_loader, scheduler=None, discovery_weight=0.3, adv_weight=0.3):
        epoch_start_time = time.time()
        model.train()
        tr_loss = 0
        loss_fn2 = nn.CrossEntropyLoss()

        for batch in tqdm(train_loader, desc='Iteration'):
            batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch)

            ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels = batch

            if self.config["adversarial"]:
                pred, pred_adv, task_pred = model(ids_sent1, segs_sent1, att_mask_sent1, position_sep)
                try:
                    half_batch_size = len(labels) // 2
                    targets, targets_adv, targets_task = labels[:half_batch_size], labels[half_batch_size:], [[0, 1]] * half_batch_size + [[1, 0]] * half_batch_size
                    targets, targets_adv, targets_task = torch.tensor(np.array(targets)).to(self.device), \
                                                        torch.tensor(np.array(targets_adv)).to(self.device), \
                                                        torch.tensor(np.array(targets_task)).to(self.device)
                except:
                    raise ValueError("batch for adversarial training has an ambiguous shape")

                loss1 = loss_fn(pred, targets.float())
                loss2 = loss_fn2(pred_adv, targets_adv.float())
                loss3 = loss_fn2(task_pred, targets_task.float())
                loss = loss1 + discovery_weight*loss2 + adv_weight*loss3
            else:
                out = model(ids_sent1, segs_sent1, att_mask_sent1, position_sep)
                if isinstance(labels, list):
                    labels = torch.tensor(np.array(labels)).to(self.device)
                loss = loss_fn(out, labels.float())

            tr_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        timing = time.time() - epoch_start_time
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Timing: {timing}, Epoch: {epoch + 1}, training loss: {tr_loss}, current learning rate {cur_lr}")

    def val(self, model, val_loader):
        model.eval()

        loss_fn = nn.CrossEntropyLoss()

        val_loss = 0
        val_preds = []
        val_labels = []
        for batch in val_loader:
            batch = tuple(t.to(self.device) for t in batch)
            ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels = batch

            with torch.no_grad():
                out = model(ids_sent1, segs_sent1, att_mask_sent1, position_sep)
                preds = torch.max(out.data, 1)[1].cpu().numpy().tolist()
                loss = loss_fn(out, labels.float())
                val_loss += loss.item()

                labels = labels.cpu().numpy().tolist()

                val_labels.extend(labels)
                if len(labels[0]) != 2:
                    for pred in preds:
                        if pred == 0:
                            val_preds.append([1,0,0])
                        elif pred == 1:
                            val_preds.append([0,1,0])
                        else:
                            val_preds.append([0,0,1])
                else:
                    val_preds.extend([[1,0] if pred == 0 else [0,1] for pred in preds])

        print(f"val loss: {val_loss}")

        val_acc, val_prec, val_recall, val_f1 = output_metrics(val_labels, val_preds)
        return val_acc, val_prec, val_recall, val_f1

    def visualize(self, model, test_dataloader, config):
        num_batches_to_plot = 50
        model.eval()

        tot_labels = None
        embeddings = None
        for i,batch in enumerate(test_dataloader):
            if i == num_batches_to_plot:
                break
            batch = tuple(t.to(self.device) for t in batch)
            ids_sent1, segs_sent1, att_mask_sent1, position_sep, labels = batch
            if tot_labels is None:
                tot_labels = labels
            else:
                tot_labels = torch.cat([tot_labels, labels], dim=0)

            with torch.no_grad():
                out = model(ids_sent1, segs_sent1, att_mask_sent1, position_sep, visualize=True)
                if embeddings is None:
                    embeddings = out
                else:
                    embeddings = torch.cat([embeddings, out], dim=0)

        SUB = str.maketrans("12", "₁₂")

        for i in range(1):
            tsne = TSNE(random_state=1)
            tsne_results = tsne.fit_transform(embeddings.detach().cpu())

            if config["visualize"] in ["student_essay", "debate"]:
                new_labels = ["Support", "Attack"]
                colors = np.array(['#035efc', '#fc9803'])
            elif config["visualize"] == "m-arg":
                new_labels = ["Support", "Attack", "Neither"]
                colors = np.array(['#035efc', '#5cfa00', '#fc9803'])
            elif config["visualize"] == "discovery":
                new_labels = ["Elaborational", "Inferential", "Contrastive"]
                colors = np.array(['#035efc', '#5cfa00', '#fc9803'])
            else:
                raise ValueError(f"The dataset {config['visualize']} cannot be plotted. Please use {config['dataset']} or discovery.")

            df_tsne = pd.DataFrame(tsne_results, columns=["x","y"])
            df_tsne["label"] = torch.argmax(tot_labels.detach(), dim=-1).cpu()

            fig, ax = plt.subplots(figsize=(8,6))
            ax.set_xlim([-80, 95])
            ax.set_ylim([-50, 50])
            ax.set_facecolor('white')
            fig.tight_layout()
            plt.xlabel('x1'.translate(SUB))
            plt.ylabel('x2'.translate(SUB))

            labels = torch.argmax(tot_labels.detach().cpu(), dim=-1).reshape(-1).numpy()

            point_colors = colors[labels]

            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=point_colors)

            legend_handles = [Patch(color=color, label=f'{label}') for i, (color, label) in enumerate(zip(colors, new_labels))]

            plt.legend(handles=legend_handles, loc='best', prop={'size': 18})

            plt.show()

            fig.savefig(f"{config['visualize']}.pdf", bbox_inches='tight')
