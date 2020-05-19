
import torch
import matplotlib.pyplot as plt
import copy

def training_loop(n_epochs, optim, model, loss_fn, dl_train, dl_val, 
                  hist=None, 
                  lr_scheduler=None,
                  device=torch.device('cpu'),
                  return_best_model=True):
    """
        return_best_model: True , the model is updated to the best model when the loop is over.
    """
    best_loss = 10000

    best_model_wts = copy.deepcopy(model.state_dict()) if return_best_model else None
    
    if hist is not None:
        best_loss = max(hist['vloss'])
    else:
        hist = {'tloss': [], 'vloss': []}
    #
    if lr_scheduler is not None:
        lr = [] # records lr after each epoch

    for epoch in range(1, n_epochs+1):
        tr_loss = 0.
        n_data = 0
        model.train() # in train mode
        for im_batch, label_batch in dl_train: # minibatch
            im_batch, label_batch = im_batch.to(device), label_batch.to(device)
            ypred = model(im_batch)
            loss_train = loss_fn(ypred, label_batch)
        
            optim.zero_grad()
            loss_train.backward()
            optim.step()
            
            # accumulate loss
            tr_loss += loss_train.item() * im_batch.shape[0]
            n_data  += im_batch.shape[0]
        # end mini-batch loop
        
        # statistics
        tr_loss /= n_data
        #
        val_loss = performance(model, loss_fn, dl_val, device)
        
        if epoch <= 5 or epoch % 1000 == 0 or epoch == n_epochs:
             print(f'Epoch {epoch}, tloss {tr_loss:.2f}   vloss {val_loss:.2f}  ')
        
        # best update
        if best_loss > val_loss:
            best_loss = val_loss
            print(f' >> best val loss updated at epoch {epoch}: {best_loss}.        ', end='\r')
            
            if return_best_model:
                best_model_wts = copy.deepcopy(model.state_dict())
        #
        
        # record for history return
        if hist is not None:
            hist['tloss'].append(tr_loss); hist['vloss'].append(val_loss) 
            
        if lr_scheduler is not None:
            lr.append(lr_scheduler.get_last_lr()) # the lr used in optim.
            lr_scheduler.step()
        # end epoch-loop
        
    if lr_scheduler is not None:
        hist['lr'] = lr
        
    # load best model weights
    if return_best_model:
        model.load_state_dict(best_model_wts)
        print(f'best model is loaded. ({best_loss})', end="")
    print ('finished training_loop(). ')
    return hist
#

def performance(model, loss_fn, dataloader, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        loss, n = 0., 0.
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            ypred = model(x)
            loss += loss_fn(ypred, y).item() * len(y)
            n += len(y)
        #
    loss /= n
    return loss
#
def plot_history(history):
    fig, axes = plt.subplots(1,1, figsize=(6,4))
    axes.set_title('Loss'); 
    axes.plot(history['tloss'], label='train'); 
    axes.plot(history['vloss'], label='val')
    axes.legend()
    return axes
#

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
