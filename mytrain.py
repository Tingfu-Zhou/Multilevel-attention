import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dldemos.Transformer.data_load import (get_batch_indices, load_cn_vocab,
                                           load_en_vocab, load_train_data,
                                           maxlen)
from dldemos.Transformer.mymodel import MATransformer # import Multilevel_Attention  model

# Config
batch_size = 64
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
n_epochs = 60
PAD_ID = 0


def main():
    device = 'cuda'
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()
    # X: en
    # Y: cn
    Y, X = load_train_data()

    print_interval = 100

    model = MATransformer(len(en2idx), len(cn2idx), PAD_ID, d_model, d_ff, # use Multilevel_Attention model
                        n_layers, heads, dropout_rate, maxlen)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    #Initializes the list of recorded losses and accuracy
    epoch_losses = []
    epoch_accuracies = []


    tic = time.time()
    cnter = 0
    for epoch in range(n_epochs):
        total_loss = 0
        total_acc = 0
        total_count = 0

        for index, _ in get_batch_indices(len(X), batch_size):
            x_batch = torch.LongTensor(X[index]).to(device)
            y_batch = torch.LongTensor(Y[index]).to(device)
            y_input = y_batch[:, :-1]
            y_label = y_batch[:, 1:]
            y_hat = model(x_batch, y_input) #bug 11.27,16:27 -step1

            y_label_mask = y_label != PAD_ID
            preds = torch.argmax(y_hat, -1)
            correct = preds == y_label
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = citerion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # Cumulative losses and accuracy
            total_loss += loss.item()
            total_acc += acc.item()
            total_count += 1

            if cnter % print_interval == 0:
                toc = time.time()
                interval = toc - tic
                minutes = int(interval // 60)
                seconds = int(interval % 60)
                print(f'{cnter:08d} {minutes:02d}:{seconds:02d}'
                      f' loss: {loss.item()} acc: {acc.item()}')
            cnter += 1

        # Calculate the average loss and accuracy
        avg_loss = total_loss / total_count
        avg_acc = total_acc / total_count
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(avg_acc)
        print(f'Epoch {epoch}: Average Loss: {avg_loss}, Average Accuracy: {avg_acc}')

    # plot the loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_accuracies, label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


    model_path = 'dldemos/Transformer/mymodel100.pth' # save the Multilevel_Attention model to a new path
    torch.save(model.state_dict(), model_path)

    print(f'Model saved to {model_path}')

    np.savez('my_model100_training_data.npz', losses=epoch_losses, accuracies=epoch_accuracies)



if __name__ == '__main__':
    main()
