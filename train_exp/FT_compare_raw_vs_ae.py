# train_exp/FT_compare_raw_vs_ae_min.py
import os, sys, math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(__file__)); sys.path.insert(0, ROOT)
import dataset as ds
from models    import MyFFNetworkForClassification
from models_ae import Autoencoder

def load_ae(ckpt):
    ck = torch.load(ckpt, map_location="cpu")
    cfg, D = ck["meta"]["config"], ck["meta"]["input_dim"]
    ae = Autoencoder(input_dim=D, enc_dims=cfg["enc_dims"], dec_dims=cfg["dec_dims"],
                     init=cfg["init"], activation=cfg["act"], use_batchnorm=cfg["bn"])
    st = ck["state"]
    for i in range(len(ae.W)): ae.W[i].copy_(st["W"][i]); ae.b[i].copy_(st["b"][i])
    if getattr(ae, "gamma", None) and "gamma" in st:
        for i in range(len(ae.gamma)):
            ae.gamma[i].copy_(st["gamma"][i]); ae.beta[i].copy_(st["beta"][i])
            ae.running_mean[i].copy_(st["running_mean"][i]); ae.running_var[i].copy_(st["running_var"][i])
    return ae, ck["meta"]

def split_fashion():
    Xtr,ytr,Xte,yte = ds.load_fashion_mnist_processed(os.path.join(ROOT,"data/processed"))
    # make a simple val split
    Xtr, ytr = Xtr.float(), torch.tensor(np.array(ytr)).long()
    Xte, yte = Xte.float(), torch.tensor(np.array(yte)).long()
    Xtr, Xva = Xtr[:-6000], Xtr[-6000:]; ytr, yva = ytr[:-6000], ytr[-6000:]
    return (Xtr,ytr),(Xva,yva),(Xte,yte)

def train_clf(model, Xtr, ytr, Xva, yva, epochs=60, lr=1e-2, batch=256, patience=10):
    best, wait = (-1.0, None), 0
    val_hist=[]
    for ep in range(1,epochs+1):
        # SGD
        idx = torch.randperm(Xtr.size(0))
        for s in range(0, Xtr.size(0), batch):
            b = idx[s:s+batch]
            logits, cache = model.forward(Xtr[b], training=True)
            loss, dY = model.loss_and_grad(logits, ytr[b])
            model.zero_grad(); model.backward(Xtr[b], cache, dY); model.step(lr)
        # val
        with torch.no_grad():
            val_logits,_ = model.forward(Xva, training=False)
            val_acc = MyFFNetworkForClassification.accuracy(val_logits, yva)
        val_hist.append(val_acc)
        if val_acc > best[0]: best = (val_acc, model.W, model.b,
                                      getattr(model,"gamma",None), getattr(model,"beta",None)); wait=0
        else: wait += 1
        if patience and wait>=patience: break
    # restore best weights
    model.W = [w.clone() for w in best[1]]; model.b = [b.clone() for b in best[2]]
    if model.use_bn:
        model.gamma=[g.clone() for g in best[3]]; model.beta=[b.clone() for b in best[4]]
    return val_hist

def main():
    (Xtr,ytr),(Xva,yva),(Xte,yte) = split_fashion()
    ae,_ = load_ae(os.path.join(ROOT,"saved_models","ae_fashion_best.pt"))
    with torch.no_grad():
        Ztr = ae.encode(Xtr); Zva = ae.encode(Xva); Zte = ae.encode(Xte)
    # same classifier/hparams for fairness
    raw = MyFFNetworkForClassification(input_dim=784, hidden_dim=256, output_dim=10,
                                       num_hidden_layers=2, init="he", activation="relu", use_batchnorm=True)
    enc = MyFFNetworkForClassification(input_dim=Ztr.shape[1], hidden_dim=256, output_dim=10,
                                       num_hidden_layers=2, init="he", activation="relu", use_batchnorm=True)
    raw_hist = train_clf(raw, Xtr, ytr, Xva, yva, epochs=60, lr=1e-2, batch=256, patience=10)
    enc_hist = train_clf(enc, Ztr, ytr, Zva, yva, epochs=60, lr=1e-2, batch=256, patience=10)

    # test set acc
    with torch.no_grad():
        ra,_ = raw.forward(Xte, training=False); raw_test = MyFFNetworkForClassification.accuracy(ra, yte)
        ea,_ = enc.forward(Zte, training=False); enc_test = MyFFNetworkForClassification.accuracy(ea, yte)
    # plots
    outd=os.path.join(ROOT,"graphs_exp"); os.makedirs(outd,exist_ok=True)
    plt.figure(figsize=(9,4)); plt.plot(raw_hist,label="raw val_acc"); plt.plot(enc_hist,label="encoded val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Validation accuracy (raw vs. encoded)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outd,"ft_val_acc_fashion.png"),dpi=150); plt.close()
    plt.figure(figsize=(4.5,4)); plt.bar(["raw","encoded"], [raw_test, enc_test])
    plt.ylim(0,1); plt.ylabel("test accuracy"); plt.title("Final test accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(outd,"ft_test_acc_fashion.png"),dpi=150); plt.close()
    print(f"RAW test_acc={raw_test:.4f} | ENCODED test_acc={enc_test:.4f}")

if __name__=="__main__": main()