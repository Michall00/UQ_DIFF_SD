# Laplace-FLARE na Stable Diffusion — Notatka

## 1. Co robimy

Kwantyfikujemy **niepewność epistemiczną** (z wag modelu) w obrazach generowanych przez Stable Diffusion v1.4, stosując podejście Laplace-FLARE z repozytorium `uqdiff` — zaadaptowane z małych modeli 1D do pełnowymiarowego text-to-image.

**Pipeline w skrócie:**

1. Generujemy referencyjne latenty $z_0$ przez DDIM (30 kroków)
2. Na parach $(z_t, \varepsilon)$ z forward diffusion fitujemy **diagonalną aproksymację Laplace'a** na ostatniej warstwie Conv2d UNeta (`conv_out`, 11 524 parametry)
3. Podczas generowania nowego obrazu, w każdym kroku DDIM obliczamy $\gamma^2_t$ i akumulujemy niepewność wzorem FLARE:

$$u_{\text{proj}} = \sum_t \left(\prod_{s>t} a_s\right)^2 \cdot b_t^2 \cdot \gamma^2_t$$

4. Wynik: obraz + mapa niepewności w przestrzeni latent (64×64×4)

## 2. Czym się różni od oryginalnego projektu

| Aspekt | Oryginał (`uqdiff`) | Nasza adaptacja (SD) |
|--------|---------------------|----------------------|
| **Model** | ScoreNet MLP (~800K params, dane 1D) | SD v1.4 UNet (~860M params, 512×512) |
| **Ostatnia warstwa** | `nn.Linear` → `laplace-torch` "out of the box" | `Conv2d(320,4,3×3)` → **ręczna implementacja** `ManualDiagLaplace` |
| **Hessian** | `laplace-torch` + `curvlinops` (automatycznie) | Ręczny diagonalny GGN via `F.unfold` patches |
| **Parametry Laplace** | ~800K (subnet/full) | **11 524** (samo conv_out = 0.001% UNeta) |
| **Obliczanie γ²** | $\gamma^2 = h^2 W_{\text{var}} + b_{\text{var}}$ (Linear) | $\gamma^2 = \text{patches}^2 \cdot W_{\text{var}} + b_{\text{var}}$ (Conv2d + `einsum`) |
| **Sampler** | DDPM (1000 kroków, bez guidance) | **DDIM** (30 kroków) + **CFG** (scale=7.5) |
| **Przestrzeń** | Pixel space (1D) | **Latent space** (64×64×4 → VAE decode → 512×512) |
| **Dane treningowe** | Syntetyczne (sinus, chirp) | Pretrained model (LAION-2B), nic nie trenujemy |

**Kluczowe różnice:**

- **`ManualDiagLaplace`** — `laplace-torch` nie obsługuje `Conv2d` jako ostatniej warstwy (rzuca `ValueError: Use model with a linear last layer`). Dlatego zaimplementowaliśmy ręcznie diagonalny GGN Hessian, który dla Conv2d wygląda tak:

$$H_{\text{diag}}[w_{c,j}] = \frac{1}{N} \sum_i \sum_l p^2_{i,j,l}$$

  gdzie $p$ to "unfolded patches" z `F.unfold` — lokalne okna receptywne konwolucji.

- **Tylko 0.001% parametrów** — to jest LLLA (Last-Layer Laplace Approximation) w najczystszej formie. Oryginał `uqdiff` mógł robić subnet albo full Hessian na całej sieci, bo była mała. My ograniczyliśmy się do jednej warstwy.

- **Latent space** — mapa $\gamma^2$ ma wymiar 64×64 (latent), nie 512×512. Upscalujemy ją biliniowo do nakładki na obraz.


## 3. Skąd wiedzieliśmy jak zrobić Laplace'a na Conv2d (diagonalny GGN)

### Pomysł: konwolucja = mnożenie macierzy na "unfoldowanych" patchach

Kluczowa obserwacja: operację `Conv2d` można sprowadzić do mnożenia macierzy, jeśli wejście "rozłożymy" na lokalne patche (tzw. **im2col trick**). Wtedy jądro konwolucji działa jak warstwa liniowa na każdym patchu osobno. A skoro tak — diagonalny GGN liczy się analogicznie jak dla `nn.Linear`, tylko na rozłożonych patchach.

W PyTorch to robi `F.unfold(features, kernel_size, padding)` — zamienia tensor $(B, C_{in}, H, W)$ na macierz patchy $(B, C_{in} \cdot k_H \cdot k_W, L)$, gdzie $L$ to liczba pozycji przestrzennych.

### Konkretnie: diagonalny GGN dla Conv2d

Dla regresji z Gaussowskim likelihood (a taki mamy — przewidujemy szum $\varepsilon$), GGN redukuje się do Fishera:

$$H = \frac{1}{N} J^\top J$$

Dla Conv2d z im2col, Jacobian względem wag jest "block-diagonal by position": każdy piksel wyjściowy zależy od jednego patcha. Diagonal GGN to wtedy:

$$H_{\text{diag}}[w_{c,j}] = \frac{1}{N} \sum_{i=1}^{B} \sum_{l=1}^{L} p^2_{i,j,l}$$

czyli suma kwadratów patchy po batchu i pozycjach przestrzennych.

Wariancja posterior to $\sigma^2 = 1 / (H_{\text{diag}} + \lambda)$, a per-pixel epistemic variance to:

$$\gamma^2 = \text{patches}^2 \cdot W_{\text{var}} + b_{\text{var}}$$

(analogon $\gamma^2 = h^2 \cdot \sigma^2_w + \sigma^2_b$ z warstwy liniowej, tyle że $h$ to "rozwinięty patch" zamiast zwykłego wektora cech).

### Źródła literaturowe

- **Im2col trick** — Chellapilla, Puri & Simard (2006), "High Performance Convolutional Neural Networks for Document Processing". Traktowanie konwolucji jako mnożenia macierzy.
- **GGN dla sieci neuronowych** — Martens (2020), "New Insights and Perspectives on the Natural Gradient Method", arXiv:[2412.16654](https://arxiv.org/abs/1412.1193). Sekcja opisująca GGN jako aproksymację Hessiana dla regresji / klasyfikacji.
- **KFAC (Kronecker-Factored Approximate Curvature)** — Martens & Grosse (2015), arXiv:[1503.05671](https://arxiv.org/abs/1503.05671). Faktoryzacja GGN dla Conv2d i FC. Diagonalny GGN jest prostszą wersją tego samego — zamiast Kronecker factorów po prostu bierzemy diagonalę.
- **Scalable Laplace** — Ritter, Botev & Barber (2018), ICLR, "A Scalable Laplace Approximation for Neural Networks". Blok-diagonalny i diagonalny GGN-Laplace dla Conv/FC.
- **Laplace Redux** — Daxberger, Kristiadi, Immer et al. (2021), NeurIPS, arXiv:[2106.14806](https://arxiv.org/abs/2106.14806). Opisuje diagonal, KFAC, i full Hessian warianty w `laplace-torch`. Sekcja 3.2 — potraktowanie Conv2d w LLLA wymaga im2col + Jacobian.

W praktyce `laplace-torch` **nie wspiera Conv2d jako ostatniej warstwy w trybie LLLA** (rzuca `ValueError`), mimo że wewnętrznie `curvlinops` potrafi liczyć GGN dla Conv2d. Dlatego musieliśmy to zrobić ręcznie — co jest proste, bo diagonalny GGN to dosłownie "suma kwadratów patchy".

## 5. Ograniczenia

- W trybie `--laplace_mode last_layer` obejmujemy tylko **conv_out** (11.5K z 860M params = 0.001%) → niepewność jest **dolnym ograniczeniem** prawdziwej epistemicznej niepewności
- W trybie `--laplace_mode subnet` obejmujemy parametry z wielu bloków UNeta, ale nadal jest to diagonalna aproksymacja oraz estymacja Monte Carlo, a nie pełny Hessian z artykułu.

## 6. Rozszerzenie: losowy subnet UNeta

Dodany został tryb `--laplace_mode subnet`, który nie ogranicza Laplace'a do `conv_out`.
Skrypt wybiera losowe tensory parametrów z różnych części UNeta (`conv_in`,
`time_embedding`, `down_blocks`, `mid_block`, `up_blocks`) i fituje diagonalny
posterior na wybranych skalarnych parametrach.

W czasie samplowania `gamma²_t` jest estymowane przez perturbacje wag:

1. próbkujemy kilka perturbacji wybranego subnetu z diagonalnego posteriora,
2. wykonujemy dodatkowe forward passy UNeta,
3. liczymy wariancję zmiany predykcji CFG `eps_theta`,
4. tę mapę traktujemy jako `diag(J Sigma J^T)` i propagujemy przez FLARE.

To jest praktyczny odpowiednik idei randomized subnetwork FLARE dla Stable
Diffusion. Nie liczymy pełnego Hessianu ani pełnych per-pixel Jacobianów, bo dla
UNeta SD byłoby to zbyt kosztowne. W zamian dostajemy network-wide signal:
niepewność zależy od parametrów z wielu bloków modelu, nie tylko od ostatniej
konwolucji.

## 7. DAAM jako attention-weighted aggregation

Dodany został osobny skrypt `run_daam_attention.py`, który używa DAAM do
wyznaczania cross-attention heatmap dla wybranych słów promptu, a następnie
może zważyć zapisane wcześniej mapy niepewności z `laplace_results.npz`.

Metryka:

$$U_{\text{DAAM}} = \sum_{x,y} A_{\text{DAAM}}(x,y) \cdot U(x,y)$$

gdzie `A_DAAM` jest znormalizowaną mapą istotności słowa/promptu, a `U` jest
mapą niepewności. Dzięki temu agregacja nie traktuje tła i prompt-relevant
regionów tak samo.

Uwaga techniczna: oficjalny pakiet DAAM pinuje starsze wersje `diffusers` i
`transformers`, dlatego jest uruchamiany przez osobne targety Makefile z
kompatybilnym środowiskiem `uv run --with daam==0.2.0 ...`, a nie jako część
głównego `stable-diffusion` extra.
