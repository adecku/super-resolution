# Super Resolution (SRCNN / EDSR / SwinIR / SRGAN)

Projekt do trenowania i ewaluacji modeli super-resolution na zbiorze DIV2K.

## 1. Wymagania

- Python 3.10+ (zalecane uruchamianie w `.venv`)
- Zainstalowane zaleznosci:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Uwaga: w `requirements.txt` nie ma wpisow `torch`/`torchvision`, wiec trzeba je doinstalowac recznie odpowiednio do platformy (CPU/CUDA/MPS).

## 2. Struktura projektu i rola plikow

### Pliki glowne

- `src/train.py` - trening modeli, zapis checkpointow, walidacja co epoke.
- `src/train_srgan.py` - trening SRGAN (pretrain generatora + faza adversarial).
- `src/config.py` - loader YAML z dziedziczeniem (`inherit:`).
- `src/datasets/div2k.py` - dataset DIV2K i DataLoadery.
- `src/utils/metrics.py` - metryki PSNR/SSIM.
- `src/utils/losses.py` - straty perceptual i adversarial uzywane przez SRGAN.
- `src/utils/device.py` - wybor urzadzenia (`cuda > mps > cpu`).
- `src/utils/seed.py` - ustawianie seedow.

### Modele

- `src/models/srcnn.py` - model SRCNN.
- `src/models/edsr.py` - model EDSR.
- `src/models/swinir.py` - model SwinIR.
- `src/models/srresnet.py` - generator SRResNet (uzywany w SRGAN).
- `src/models/srgan_discriminator.py` - discriminator SRGAN.
- `src/models/factory.py` - wspolny factory modeli (train/eval/benchmark).

### Skrypty pomocnicze

- `scripts/eval_model.py` - ewaluacja checkpointu modelu na walidacji.
- `scripts/eval_bicubic_baseline.py` - baseline bicubic.
- `scripts/benchmark_inference.py` - benchmark czasu inferencji.
- `scripts/collect_results.py` - agregacja metryk do `outputs/results.csv` i `outputs/results.json`.
- `scripts/sanity_check_div2k.py` - zapis przykladow LR/HR/bicubic do szybkiej kontroli danych.

### Konfiguracje

- `configs/common.yaml` - wspolne ustawienia.
- `configs/train_srcnn_x2.yaml`, `configs/train_srcnn_x4.yaml`
- `configs/train_edsr_x2.yaml`, `configs/train_edsr_x4.yaml`
- `configs/train_swinir_x2.yaml`, `configs/train_swinir_x4.yaml`
- `configs/train_srgan_x2.yaml`, `configs/train_srgan_x4.yaml`

### Testy

- `test_imports.py` - szybki smoke test importow, metryk i konfiguracji.

## 3. Oczekiwana struktura danych DIV2K

W `configs/common.yaml` domyslnie:

- `paths.data_root: data/raw/DIV2K`

Wymagane katalogi (x2 i x4):

- `DIV2K_train_HR`
- `DIV2K_train_LR_bicubic/X2`
- `DIV2K_train_LR_bicubic_X4/X4`
- `DIV2K_valid_HR`
- `DIV2K_valid_LR_bicubic/X2`
- `DIV2K_valid_LR_bicubic_X4/X4`

## 4. Jak uruchomic

## 4.1 Smoke test srodowiska

```bash
source .venv/bin/activate
python test_imports.py
```

## 4.2 Trening

Przyklady:

```bash
# SRCNN x2
python src/train.py --config configs/train_srcnn_x2.yaml

# EDSR x4
python src/train.py --config configs/train_edsr_x4.yaml

# SwinIR x2
python src/train.py --config configs/train_swinir_x2.yaml
```

Wyniki treningu trafiaja do:

- `outputs/runs/<model>_x<scale>/<timestamp>/`

W katalogu run:

- `last.pth` - ostatni checkpoint,
- `best.pth` - najlepszy wg PSNR,
- `ckpt_ep{N}.pth` - checkpoint per epoka,
- `config_resolved.yaml` - config po dziedziczeniu,
- logi TensorBoard i przykladowe obrazy.

## 4.2.1 Trening SRGAN

SRGAN ma osobny skrypt treningowy i 2 fazy:

- `pretrain` generatora (rekonstrukcja),
- `adversarial` (generator + discriminator).

Przyklady:

```bash
# SRGAN x2
python src/train_srgan.py --config configs/train_srgan_x2.yaml

# SRGAN x4
python src/train_srgan.py --config configs/train_srgan_x4.yaml
```

Wyniki treningu trafiaja do:

- `outputs/runs/srgan_x<scale>/<timestamp>/`

## 4.3 Wznowienie treningu

```bash
python src/train.py \
  --config configs/train_srcnn_x2.yaml \
  --resume outputs/runs/srcnn_x2/<timestamp>/last.pth

# SRGAN (resume)
python src/train_srgan.py \
  --config configs/train_srgan_x2.yaml \
  --resume outputs/runs/srgan_x2/<timestamp>/last.pth
```

## 4.4 Walidacja modelu (po checkpointcie)

```bash
python scripts/eval_model.py \
  --config configs/train_srcnn_x2.yaml \
  --ckpt outputs/runs/srcnn_x2/<timestamp>/best.pth
```

Skrypt zapisuje metryki i obrazy do:

- `outputs/figures/eval_<model>_x<scale>/`

Dla SRGAN uzywasz tego samego skryptu, np.:

```bash
python scripts/eval_model.py \
  --config configs/train_srgan_x2.yaml \
  --ckpt outputs/runs/srgan_x2/<timestamp>/best.pth
```

## 4.5 Baseline bicubic

```bash
python scripts/eval_bicubic_baseline.py --config configs/train_srcnn_x2.yaml
```

Wynik:

- `outputs/figures/bicubic_baseline_x<scale>.json`

## 4.6 Benchmark inferencji

```bash
python scripts/benchmark_inference.py \
  --config configs/train_srcnn_x2.yaml \
  --ckpt outputs/runs/srcnn_x2/<timestamp>/best.pth \
  --num-batches 10 \
  --warmup-batches 2

# SRGAN (benchmark generatora)
python scripts/benchmark_inference.py \
  --config configs/train_srgan_x2.yaml \
  --ckpt outputs/runs/srgan_x2/<timestamp>/best.pth \
  --num-batches 10 \
  --warmup-batches 2
```

## 4.7 Agregacja wynikow

```bash
python scripts/collect_results.py
```

Wyniki zbiorcze:

- `outputs/results.csv`
- `outputs/results.json`

## 4.8 Wizualny sanity-check danych

```bash
python scripts/sanity_check_div2k.py \
  --config configs/train_srcnn_x2.yaml \
  --out outputs/figures/sanity_check_srcnn_x2
```

## 5. TensorBoard

```bash
tensorboard --logdir outputs/runs
```

Potem otworz adres z terminala (zwykle `http://localhost:6006`).

## 6. Najczestsze problemy

- `No module named torch` / `yaml` / `numpy`:
  uruchamiasz poza `.venv` albo brakuje zaleznosci.
- `LR folder does not exist`:
  sprawdz strukture `data/raw/DIV2K` i skale x2/x4.
- `Model architecture mismatch` przy `eval`:
  uzyj configu zgodnego z checkpointem treningowym.
- SRGAN i PSNR/SSIM:
  SRGAN czesto daje lepsza jakosc wizualna, ale nie zawsze najlepszy PSNR/SSIM.
- SRGAN i VGG perceptual loss:
  przy `vgg_pretrained: true` torchvision moze probowac pobrac wagi VGG19; na maszynie offline ustaw `train.srgan.vgg_pretrained: false`.
