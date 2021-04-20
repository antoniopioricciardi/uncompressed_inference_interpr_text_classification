import time
import torch
import numpy as np


# TODO: Could be a general class (not text-specific)
class Trainer:
    def __init__(self, model, train_dataset, test_dataset, n_epochs, model_path):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs
        self.model_path = model_path

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            start_time = time.time()
            tot_train_len = 0
            tot_val_len = 0
            train_loss = 0
            train_acc = 0
            val_acc = 0

            for batch in self.train_dataset:
                self.model.zero_grad()
                tot_train_len += len(batch)
                data = torch.from_numpy(np.array([np.array(el[0]) for el in batch])).to(self.model.device)
                labels = torch.tensor([el[1] for el in batch]).to(self.model.device)
                pred = self.model(data.float())
                # print(pred.argmax(1), '-', labels)

                loss = self.model.loss(pred, labels)
                train_loss += loss
                train_acc += (pred.argmax(1) == labels).sum().item()

                loss.backward()
                self.model.optimizer.step()

            with torch.no_grad():
                for batch in self.test_dataset:
                    tot_val_len += len(batch)
                    data = torch.from_numpy(np.array([np.array(el[0]) for el in batch])).to(self.model.device)
                    labels = torch.tensor([el[1] for el in batch]).to(self.model.device)
                    pred = self.model(data.float())
                    val_acc += (pred.argmax(1) == labels).sum().item()
            # with torch.no_grad():
            #     for batch in batches_list_val:
            #         tot_val_len += len(batch)
            #         data = torch.from_numpy(np.array([np.array(el[0]) for el in batch])).to(model.device)
            #         labels = torch.tensor([el[1] for el in batch]).to(model.device)
            #         pred = model(data.float())
            #         val_acc += (pred.argmax(1) == labels).sum().item()
            #     # Adjust the learning rate
            # # model.scheduler.step()

            train_loss = train_loss / tot_train_len
            train_acc = train_acc / tot_train_len
            val_acc = val_acc / tot_val_len

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
            print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
            # print(f'\tLoss: {val_acc:.4f}(valid)\t|\tAcc: {val_acc * 100:.1f}%(valid)')
            print(f'\tAcc: {val_acc * 100:.1f}%(valid)')
            torch.save(self.model.state_dict(), self.model_path)  # + 'model.pth')
            # torch.save(self.model.optimizer.state_dict(), self.model_path + 'optimizer.pth')

        return self.model
