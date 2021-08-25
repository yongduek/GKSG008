# training_loop.py
#
# yndk@sogang.ac.kr
#
# training loop function for classification

import time
import torch
import matplotlib.pyplot as plt
import copy

def training_loop(n_epochs, optim, model, loss_fn, dl_train, dl_val, 
                  hist=None, 
                  lr_scheduler=None,
                  lr_scheduler_step_update_flag=False,
                  device=torch.device('cpu'),
                  return_best_model=True,
                  verbose=False):
    """
        return_best_model: True , the model is updated to the best model when the loop is over.
    """
    best_acc = 0

    best_model_wts = copy.deepcopy(model.state_dict()) if return_best_model else None
    
    if hist is not None:
        best_acc = max(hist['vacc'])
    else:
        hist = {'tloss': [], 'tacc': [], 'vloss': [], 'vacc': []}
    #
    if lr_scheduler is not None:
        lr = [] # records lr after each epoch or step, according to lr_scheduler_step_update_flag

    flag_overfit, flag_break = False, False
        
    for epoch in range(1, n_epochs+1):
        start_time = time.time()
        
        tr_loss, tr_acc = 0., 0.
        n_data = 0
        model.train() # in train mode
        for im_batch, label_batch in dl_train: # minibatch
            im_batch, label_batch = im_batch.to(device), label_batch.to(device)
            ypred = model(im_batch)
            loss_train = loss_fn(ypred, label_batch)
        
            optim.zero_grad()
            loss_train.backward()
            optim.step()

            if lr_scheduler is not None and lr_scheduler_step_update_flag == True:
                lr.append(lr_scheduler.get_last_lr()) # the lr used in optim. From torch 1.5, get_lr() -> get_last_lr()
                lr_scheduler.step()
            
            # accumulate correct prediction
            tr_acc  += (torch.argmax(ypred.detach(), dim=1) == label_batch).sum().item() # number of correct predictions
            tr_loss += loss_train.item() * im_batch.shape[0]
            n_data  += im_batch.shape[0]
        # end mini-batch loop
        
        # statistics
        tr_loss /= n_data
        tr_acc  /= n_data
        #
        val_loss, val_acc = performance(model, loss_fn, dl_val, device)
        
        if (epoch <= 5 or epoch % 1000 == 0 or epoch == n_epochs) or verbose:
            ellapsed = time.time() - start_time
            print(f'Epoch {epoch}, tloss {tr_loss:.2f} vloss {val_loss:.2f}  t_acc: {tr_acc:.2f} v_acc: {val_acc:.2f} | ellapsed: {ellapsed:.1f}')
        
        # best accuracy
        if best_acc < val_acc:
            best_acc = val_acc
            print(f'>> best val accuracy updated at epoch {epoch}: {best_acc} / (tacc: {tr_acc}) --', end='\r')
            
            if return_best_model:
                best_model_wts = copy.deepcopy(model.state_dict())
        #
        if tr_acc > 0.99:
            print(f'warning: training accuracy very large, must be in overfitting now  {tr_acc}')
            if flag_overfit == False:
                flag_overfit = True
            else:
                flag_break = True
        # record for history return
        if hist is not None:
            hist['tloss'].append(tr_loss); hist['vloss'].append(val_loss) 
            hist['tacc'].append(tr_acc);   hist['vacc'].append(val_acc)
            
        if lr_scheduler is not None and lr_scheduler_step_update_flag == False:
            lr.append(lr_scheduler.get_last_lr()) # the lr used in optim. From torch 1.5, get_lr() -> get_last_lr()
            lr_scheduler.step()
        # end epoch-loop
        
        if flag_break:
            break  # stop traing loop
            
    if lr_scheduler is not None:
        hist['lr'] = lr
        
    # load best model weights
    if return_best_model:
        model.load_state_dict(best_model_wts)
        print('best model is loaded. ', end="")
    print ('finished training_loop(). ')
    return hist
#

def performance(model, loss_fn, dataloader, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        loss, acc, n = 0., 0., 0.
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            ypred = model(x)
            loss += loss_fn(ypred, y).item() * len(y)
            p = torch.argmax(ypred, dim=1)
            acc += (p == y).sum().item()
            n += len(y)
        #
    loss /= n
    acc /= n
    return loss, acc
#
def plot_history(history, skip=0, text=' '):
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    axes[0].set_title('Loss    ' + text); 
    axes[0].plot(history['tloss'][skip:], label='train'); 
    axes[0].plot(history['vloss'][skip:], label='val')
    axes[0].legend()
    
    max_vacc = max(history['vacc'])
    axes[1].set_title(f'Acc. vbest: {max_vacc:.2f}   ' + text)
    axes[1].plot(history['tacc'][skip:], label='train'); 
    axes[1].plot(history['vacc'][skip:], label='val')
    axes[1].legend()
#

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
