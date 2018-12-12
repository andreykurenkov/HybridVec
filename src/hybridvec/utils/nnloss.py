from torch.autograd import Variable

def get_loss_nll(acc_loss, norm_term):
    if isinstance(acc_loss, int):
        return 0
    
    # total loss for all batches
    loss = acc_loss.data
    loss /= norm_term
    loss =  (Variable(loss).data)[0]
    return loss

