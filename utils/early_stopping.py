import torch
import numpy as np 

class EarlyStopping():
    def __init__(self,patience=0,verbose=True,path='check_point.pt'):
        """[summary]
        Args:
            patience (int): 몇 epoch 만큼 계속해서 오차가 증가하면 학습을 중단할지 결정한다.
            verbose (bool): validation loss log 를 보여줄지 결정한다.
            path (str, optional): model.pt 를 어디에 저장할지 결정한다.
        """        

        self.patience=patience
        self.verbose=verbose
        self._path=path
        self._step=0
        self._min_val_loss=np.inf
        self._early_stopping=False
    

    def __call__(self,val_loss,model):
        if self._early_stopping: return 

        if self._min_val_loss < val_loss:  #val_loss 증가 
            if self._step >=self.patience:
                self._early_stopping=True
                if self.verbose:
                    print(f'Validation loss increased for {self.patience} epochs...\t Best_val_loss : {self._min_val_loss}')
            elif self._step<self.patience:
                self._step+=1
        else:
            self._step=0
            if self.verbose:
                print(f'Validation loss decreased ({self._min_val_loss:.6f} ---> {val_loss:.6f})\tSaving model..."{self.path}"')
            self._min_val_loss=val_loss
            self.save_checkpoint(model)

    def save_checkpoint(self,model):
        torch.save(model.state_dict(),self.path)

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self,path):
        self._path=path

