import torch
import tqdm
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

def testModel(model,dataloaders,device):
    model.eval()

    zerosamples = 0
    zerocorrect = 0
    onesamples = 0
    onecorrect = 0
    num_correct = 0
    num_samples = 0

    y_true = torch.empty(0, dtype=torch.int).to(device)
    y_pred = torch.empty(0, dtype=torch.int).to(device)

    loader = dataloaders["testing"]

    with torch.no_grad():
        with tqdm.tqdm(loader, unit="batch") as tbatch:
            for batch_idx, (x, y) in enumerate(tbatch):
                tbatch.set_description(f"Batch {batch_idx}")
                
                if x.shape[2] <= 1536 and x.shape[3] <= 1536:
                    x = x.to(device=device)
                    y = y.to(device=device)
                    y_true = torch.cat((y_true, y))
                    
                    scores = model(x)

                    # since verdeoliva net has a bit of strange outputs, we flatten 
                    # them first
                    scores = torch.squeeze(scores,dim=2)
                    scores = torch.squeeze(scores,dim=2)

                    _, predictions = scores.max(1)
                    y_pred = torch.cat((y_pred, predictions))

                    zerosamples += len(y[y==0])
                    zerocorrect += (y[y==predictions]==0).sum().item()
                    onesamples += len(y[y==1])
                    onecorrect += (y[y==predictions]==1).sum().item()
                    num_samples += predictions.size(0)
                    num_correct += (predictions == y).sum()
                    
                    zeroaccuracy = 0
                    if zerosamples > 0:
                        zeroaccuracy = float(zerocorrect)/float(zerosamples)*100
                    oneaccuracy = 0
                    if onesamples > 0:
                        oneaccuracy = float(onecorrect)/float(onesamples)*100
                    accuracy = float(num_correct)/float(num_samples)*100
                    tbatch.set_postfix(accuracy=accuracy, real=zeroaccuracy, fake=oneaccuracy)

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
        # do not create dataFrame if there are nan in the cf_matrix
        if cf_matrix.shape == (2,2):
            df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = [i for i in ['real','fake']],
                         columns = [i for i in ['real','fake']])
            print('Confusion_Matrix:\n {}\n'.format(df_cm))

        print(f'Got tot: {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} \n')
        if zerosamples > 0:
            print(f'Got Real: {zerocorrect} / {zerosamples} with accuracy {float(zerocorrect)/float(zerosamples)*100:.2f} \n')
        if onesamples > 0:
            print(f'Got Fake: {onecorrect} / {onesamples} with accuracy {float(onecorrect)/float(onesamples)*100:.2f} \n')

def test_on_folder(model,folder_dataloader,transforms,device):
    accuracy = 0

    model.to(device)
    model.eval()

    num_correct = 0
    num_samples = len(folder_dataloader)
    with torch.no_grad():
        with tqdm.tqdm(folder_dataloader, unit="batch") as tbatch:
            for batch_idx, (x, y) in enumerate(tbatch):
                x = x.to(device)
                y = y.to(device)

                scores = model(x)
                _ , prediction = scores.max(1)
                if prediction == y.item():
                    num_correct += 1

                print(str(folder_dataloader.dataset.samples[batch_idx]) + "label: " + str(y.item()) + "predicted: " + str(prediction))

    accuracy = num_correct / num_samples

    return accuracy