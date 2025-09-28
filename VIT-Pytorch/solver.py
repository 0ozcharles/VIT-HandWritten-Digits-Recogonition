import os
import torch
import torch.nn as nn
from torch import optim
from model import VisionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader import get_loader


class Solver(object):
    def __init__(self, args):
        self.args = args
        self.train_loader=args.train_loader
        self.test_loader = args.test_loader
        self.model = VisionTransformer(args).cuda()
        self.ce = nn.CrossEntropyLoss()

        print('--------Network--------')
        print(self.model)

        if args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'Transformer.pt')))

    def test_dataset(self, db='test'):
        self.model.eval()

        actual = []
        pred = []

        if db.lower() == 'train':
            loader = self.train_loader
        elif db.lower() == 'test':
            loader = self.test_loader

        for (imgs, labels) in loader:
            imgs = imgs.cuda()

            with torch.no_grad():
                class_out = self.model(imgs)
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args.n_classes))

        return acc, cm

    def test(self):
        train_acc, cm = self.test_dataset('train')
        print("Tr Acc: %.2f" % train_acc)
        print(cm)

        test_acc, cm = self.test_dataset('test')
        print("Te Acc: %.2f" % test_acc)
        print(cm)

        return train_acc, test_acc

    def train(self, return_metrics=False):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(self.model.parameters(), self.args.lr, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epochs)

        train_loss_list = []
        val_acc_list = []

        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0

            for i, (imgs, labels) in enumerate(self.train_loader):
                imgs, labels = imgs.cuda(), labels.cuda()

                logits = self.model(imgs)
                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                total_loss += clf_loss.item()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (
                        epoch + 1, self.args.epochs, i + 1, iter_per_epoch, clf_loss.item()))

            avg_loss = total_loss / iter_per_epoch
            train_loss_list.append(avg_loss)

            test_acc, _ = self.test_dataset('test')
            val_acc_list.append(test_acc)
            print("Test acc: %0.2f" % test_acc)

            cos_decay.step()

        if return_metrics:
            return train_loss_list, val_acc_list
