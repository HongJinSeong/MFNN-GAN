from math import cos, pi

# loss를 cosine decay 시키기 위함
# cyclic 없이 오로지 감소만 시키는 목적으로 구현 (cyclic 고려 X)

class CosineDecay_for_loss():
    def __init__(self,
                 init_decay_epochs,
                 T):
        # decay 주기
        self.T=T
        # decay 시작 Epoch
        self.init_decay_epochs = init_decay_epochs
        #현재 decay rate
        self.now_decay_rate = 0
        #decay 시킨 횟수
        self.decay_iter_num = 1
        

    def step(self,epoch,loss):
        if epoch > self.init_decay_epochs and self.init_decay_epochs+self.T>epoch:
             self.now_decay_rate = ((1 + cos(pi * epoch / self.T)) / 2)
             loss = loss * self.now_decay_rate
        elif epoch <self.init_decay_epochs:
             return loss
        else :
             loss=loss*self.now_decay_rate

        return loss

    def get_decay_rate(self):
        return self.now_decay_rate