# Neural Network Trainer (NumPy + PyQt5)

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![GUI PyQt5](https://img.shields.io/badge/GUI-PyQt5-41cd52.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightgrey.svg)

Rete neurale feedforward (MLP) scritta in NumPy per la classificazione su MNIST, Fashion-MNIST e CIFAR-10, con interfaccia grafica (PyQt5) per l’addestramento, la visualizzazione dell’architettura, l’andamento delle metriche e un’anteprima delle predizioni.

---

## Caratteristiche principali

- Dataset supportati con download automatico e verifica checksum MD5:
  - MNIST, Fashion-MNIST (formato IDX compresso)
  - CIFAR-10 (archivio tar.gz) con estrazione sicura
- Rete MLP completamente connessa: `Dense → ReLU → (Dropout) → ... → Dense → Softmax`
  - Inizializzazione He per gli strati densi
  - Perdita Cross-Entropy, Softmax colonnare, mini-batch training
  - Dropout opzionale sugli strati nascosti
- Ottimizzatori integrati (puramente NumPy):
  - SGD, SGD con Momentum, Adam, AdamW (con weight decay)
- Interfaccia grafica (PyQt5):
  - Configurazione esperimenti (dataset, ottimizzatore, LR, batch size, epochs, dropout, dimensioni hidden, seed, limiti train/test, intervallo di valutazione)
  - Vista dell’architettura (grafo dei layer con nodi e operazioni)
  - Grafici live di Loss/Accuracy (train e test)
  - Galleria delle prime 100 predizioni sul test set (con evidenza corretto/errato)
  - Log in tempo reale e barra di progresso
- CLI equivalente per esecuzioni da terminale

---

## Requisiti

- Python 3.9+ (consigliato)
- Dipendenze Python:
  - numpy
  - matplotlib
  - PyQt5 (per la GUI)

Installa tutto con:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Struttura del progetto

- `constants.py` — iperparametri di default (dimensioni layer, LR, batch size, epochs, ecc.).
- `dataset_downloader.py` — utilità per download, verifica MD5, estrazione (sicura) e caricamento in memoria di MNIST/Fashion-MNIST/CIFAR-10. Restituisce tensori nel formato atteso dalla rete.
- `utils.py` — funzioni matematiche di base: ReLU, derivata ReLU, Softmax, one-hot.
- `layer.py` — implementazione dei layer:
  - `DenseLayer` (He init, forward/backward, memorizza gradienti dW/db)
  - `ReLULayer`, `DropoutLayer`, `SoftmaxLayer`
  - `CrossEntropyLoss`
- `optimizer.py` — ottimizzatori NumPy: `SGD`, `SGDMomentum`, `Adam`, `AdamW`.
- `neural_network.py` — composizione della rete, forward/backward, loss, train step, accuracy.
- `main.py` — interfaccia a riga di comando (CLI) per il training.
- `gui.py` — applicazione PyQt5 per il training interattivo.
- `data/` — cache locale dei dataset scaricati (creata automaticamente).
- `run.bat` — esempio di esecuzione CLI preconfigurata per Windows.

---

## Dati e formati (contratto I/O)

- Le immagini sono normalizzate in `[0,1]` (float32).
- Formati tensori principali:
  - `X`: shape `(D, N)` dove `D` = numero di feature (784 per MNIST/Fashion-MNIST, 3072 per CIFAR-10), `N` = numero di esempi.
  - `Y` (one-hot): shape `(K, N)` dove `K` = classi (10). Necessario per `train_step`.
  - `y` (etichette intere): shape `(N,)`, usato per `accuracy` e preview.
- Split supportati: `train` e `test`.
- Opzioni caricamento: normalizzazione abilitata di default, etichette one-hot opzionali, parametro `limit` per sottocampionare (utile per prove rapide).

---

## Architettura della rete

- Sequenza generata a partire da `layer_sizes = [n_in, h1, h2, ..., n_out]`:
  - Per ogni transizione `n_in → h`: `Dense(h)` → `ReLU` → (opzionale `Dropout`)
  - Uscita: `Softmax`
- Inizializzazione He per gli strati densi (`W ~ N(0, \sqrt{2/d_in})`, `b=0`).
- Loss: `CrossEntropyLoss` (con clipping numerico interno).
- Backprop: gradienti da CrossEntropy+Softmax, poi propagazione inversa e `optimizer.step()`.

Edge cases considerati:
- `dropout_rate = 0.0` disabilita il dropout senza overhead.
- Batch vuoto o limiti dataset inconsistenti generano errori chiari.
- Estrazione CIFAR-10 con protezione percorsi (no path traversal).

---

## Ottimizzatori supportati

- `SGD(lr)`
- `SGDMomentum(lr, beta=0.9)`
- `Adam(lr, beta1=0.9, beta2=0.999, eps=1e-8)`
- `AdamW(lr, ..., weight_decay=1e-2)`

Gli ottimizzatori operano automaticamente su tutti i layer addestrabili (`DenseLayer`) presenti nella rete.

---

## Modalità d’uso

### GUI (consigliata per esplorazione)

Avvia l’applicazione grafica:

```powershell
python gui.py
```

Funzionalità principali:
- Sezione Configurazione: dataset, ottimizzatore, epochs, batch size, learning rate, dropout, hidden layers (es. `256,128,64`), limiti train/test, intervallo valutazione, seed.
- Controlli: Avvia/Ferma + barra di avanzamento + log.
- Metriche: grafici Loss, Accuracy (train/test) aggiornati a ogni epoch.
- Predizioni: galleria dei primi 100 esempi del test set con confidenza e indicazione corretto/errato.

### CLI (scriptabile e riproducibile)

Mostra l’help:

```powershell
python main.py --help
```

Parametri principali:
- `--dataset {mnist,fashion_mnist,cifar10}` (default: `mnist`)
- `--epochs INT` (default da `constants.py`)
- `--batch-size INT`
- `--optimizer {SGD,SGDMomentum,Adam,AdamW}`
- `--lr FLOAT`
- `--dropout FLOAT` (in `[0,1)`, applicato agli strati hidden)
- `--hidden H1 H2 ...` (es. `--hidden 256 128 64`)
- `--train-limit INT`, `--test-limit INT` (sottocampionamento)
- `--eval-interval INT` (0 = disabilita)
- `--seed INT`

Esempi:

```powershell
# MNIST con Adam
python main.py --dataset mnist --optimizer Adam --epochs 10 --batch-size 128 --lr 0.001 --hidden 128 64

# CIFAR-10 con AdamW e dropout
python main.py --dataset cifar10 --optimizer AdamW --epochs 50 --batch-size 128 --lr 0.001 --dropout 0.1 --hidden 256 128 64

# Esecuzione preconfigurata (Windows)
./run.bat
```

I dataset vengono scaricati in `data/<dataset>/` al primo utilizzo (cache locale persistente).

---

## Consigli pratici e prestazioni

- CIFAR-10 è più impegnativo: inizia con pochi epoch e hidden layer piccoli (es. `128,64`) e batch size moderato (128-256).
- `SGD` è semplice ma potrebbe convergere lentamente; `SGDMomentum`/`Adam`/`AdamW` tendono a stabilizzare e accelerare.
- `dropout` aiuta a regolarizzare su modelli con molti neuroni; parti da `0.1`-`0.2`.
- Imposta `--train-limit` per prototipare rapidamente.