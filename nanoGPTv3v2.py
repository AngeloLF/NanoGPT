import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import numpy as np
# from time import perf_counter as ptime
from time import time as ptime
from time import sleep
from classAeTime import AeTime
from classInfo import StatGPT
import matplotlib.pyplot as plt
import os
import sys

"""
Ce fichier est le fichier central du projet. Il y est codé from scratch un modèle réduit de GPT, que l'on a appelle NanoGPT
"""


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, block_size, n_embd, modelDropout):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(modelDropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, block_size, n_embd, modelDropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embd, modelDropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(modelDropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, modelDropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(modelDropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, modelDropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embd, modelDropout)
        self.ffwd = FeedFoward(n_embd, modelDropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.batch_size    = kwargs['batch_size']
        self.block_size    = kwargs['block_size']
        self.learning_rate = kwargs['learning_rate']
        self.max_iters     = kwargs['max_iters']
        self.eval_interval = kwargs['eval_interval']
        self.eval_iters    = kwargs['eval_iters']
        self.n_embd        = kwargs['n_embd']
        self.n_head        = kwargs['n_head']
        self.n_layer       = kwargs['n_layer']
        self.modelDropout  = kwargs['modelDropout']

        self.seed          = kwargs['seed']

        self.inputData     = kwargs['inputData']
        self.outputData    = kwargs['outputData']

        self.limitLearning = kwargs['limitLearning'] # Pour des tests

        # ------------

        
        self.stat = StatGPT(kwargs, saveFreq=10, termSave=kwargs['termSave'])
        self.aetime = AeTime()
        self.ptime = None

        # ------------

        torch.manual_seed(self.seed)

        self.aetime.beg('Init')

        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open(self.inputData, 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        self.n = int(0.9*len(data)) # first 90% will be train, rest val
        self.train_data = data[:self.n]
        self.val_data = data[self.n:]

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.Sequential(*[Block(self.n_embd, n_head=self.n_head, block_size=self.block_size, modelDropout=self.modelDropout) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

        self.aetime.end('Init')

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T)) #, device=device)) ***** # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, showStep=False, seed=None):

        if seed != None:
            torch.manual_seed(seed)

        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # if showStep:
            #     print(f"IDX COND : {self.decode(idx_cond[0].tolist())}")
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
            if showStep:
                # permet d'afficher lettre par lettre (juste esthétique)
                sys.stdout.write(self.decode(idx_next[0].tolist()))
                sys.stdout.flush()

        return idx


    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        # x, y = x.to(device), y.to(device) *****
        return x, y


    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.aetime.beg('esti EVAL')
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.aetime.end('esti EVAL')
        self.train()
        return out


    def letsLearning(self):

        # m = model.to(device) *****
        # print the number of parameters in the model
        # print(sum(p.numel() for p in self.parameters())/1e6, 'M parameters') # *****
        nbParam = sum(p.numel() for p in self.parameters())
        self.stat.nbParam = nbParam

        # create a PyTorch optimizer
        self.ptime = ptime()

        self.aetime.beg('Create Opti')
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.aetime.end('Create Opti')
        # iter = -1
        for iter in range(self.max_iters):                 # TEST LIMIT DE TEMPS (a enlever)
        # while ptime() - self.ptime < self.limitLearning: # TEST LIMIT DE TEMPS (a ajouter)
        #     iter += 1                                    # TEST LIMIT DE TEMPS (a ajouter)

            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0: # or iter == self.max_iters - 1:
                self.aetime.beg('estimate_loss')

                losses = self.estimate_loss()
                tm = (ptime() - self.aetime.total)/60
                print(f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f} [DIFF : {losses['val']-losses['train']:.3f}][*$ {tm:.1f} min]")

                self.stat.stepTV_loss.append(iter)
                self.stat.timeTV_loss.append(ptime() - self.ptime)
                self.stat.train_loss.append(losses['train'])
                self.stat.value_loss.append(losses['val'])

                self.aetime.end('estimate_loss')


            self.aetime.beg('rest')
            # sample a batch of data
            xb, yb = self.get_batch('train')
            # evaluate the loss
            logits, loss = self(xb, yb)


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter % self.stat.sf == 0:
                self.stat.time.append(ptime() - self.ptime)
                self.stat.loss.append(loss)
            self.aetime.end('rest')

        
        self.aetime.show()
        self.stat.aetime = self.aetime
        self.stat.saveStat()


    def generateFromContext(self, context='\n', new_token=100, showStep=False, seed=None):

        contextEncode = torch.tensor([self.encode(context)], dtype=torch.long)

        if showStep:
            print('Generation : ')
            print(self.decode(contextEncode[0].tolist()), end="")

        msgFinal = self.decode(self.generate(contextEncode, max_new_tokens=new_token, showStep=showStep, seed=seed)[0].tolist()) # replace model -> m *****

        if showStep:
            print('\n')

        return msgFinal

    def saveGPT(self):

        with open(self.outputData, 'wb') as f:
            pickle.dump(self, f)


    def plotLoss(self):

        plt.plot(self.step, self.loss, color='k', alpha=0.8, label='Loss Loop')
        plt.plot(self.stepTV_loss, self.train_loss, color='r', label='Train loss')
        plt.plot(self.stepTV_loss, self.value_loss, color='g', label='Val. loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()

        plt.plot(self.stepTV_loss, np.array(self.train_loss) / np.array(self.value_loss), color='r')
        plt.title('Suivi du rapport train / val loss')
        plt.show()

        plt.plot(self.time, self.loss, color='k', alpha=0.8, label='Loss Loop' )
        plt.plot(self.timeTV_loss, self.train_loss, color='r', label='Train loss')
        plt.plot(self.timeTV_loss, self.value_loss, color='g', label='Val. loss')
        plt.title('Suivi du loss en fonction du temps')
        plt.show()
                