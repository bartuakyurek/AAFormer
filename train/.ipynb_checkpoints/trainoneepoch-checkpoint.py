#######################################################################################
# 
#    The code given on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
#    is used and modified according to the project
#
#######################################################################################

from train.plot import image_plot


def train_one_epoch(model, optimizer, epoch_index, dataloader_trn, loss_fn, loss_batch, tb_writer, device):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch in enumerate(dataloader_trn):
        print(f"iteration {i+1}")

        # STEP 0: Get query image and support images with corresponding masks
        query_img = batch['query_img'].to(device)
        query_mask = batch['query_mask'].to(device)
        supp_imgs = batch['support_imgs'].to(device)
        supp_masks = batch['support_masks'].to(device)
                
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # STEP 1: Get predicted mask
        #try:
        outputs = model(query_img, supp_imgs, supp_masks, normalize=True) # TODO: make normalize false if torch.CrossEntropy is used
        #except:
        #    print(">> Unexpected error occured, skipping this batch...")
        #    continue
        
        # STEP 2: Compute Dice loss
        loss = loss_fn(outputs, query_mask)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % loss_batch == loss_batch - 1:
            last_loss = running_loss / loss_batch # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader_trn) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            
            image_plot(query_img=query_img, query_mask=query_mask, preds=outputs)
            
    return last_loss