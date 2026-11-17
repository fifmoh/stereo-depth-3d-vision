# Sujet 23 — Introduction to Depth Estimation and 3D Vision
## Plan de construction (à valider avant exécution)

> Statut : **planification** — aucun code d'implémentation écrit pour l'instant.
> Contrainte cadre : pipeline **classique, CPU-only**, sans GPU, sans API payante, sans compte sur un portail de datasets.
> Source du sujet : `Cahiers_de_Charges_Pipelines_Complets.docx`, Sujet 23 (Middlebury Stereo / KITTI Stereo).

**Décisions verrouillées** :
- Dataset principal : **Middlebury Stereo 2014** (vérité terrain dense + `calib.txt`), KITTI uniquement en secondaire optionnel.
- **Langues** : README / docstrings / code / identifiants → **anglais** ; **rapport (+ notebooks commentés) → français**. Aucun mélange dans un même fichier.
- **Pipeline 100 % classique** (§1.4) : StereoBM/SGBM + calibration + grille de paramètres + triangulation. **Pas de RandomForest de fiabilité** (option bonus écartée).
- **Phase 1 réelle** : calibration damier + `stereoCalibrate` / `stereoRectify` effectivement implémentée (pas seulement la vérification `dy≈0`).
- **Livrables de démo** : **CLI + notebooks** ; UI à slider envisagée seulement si le temps le permet (cf. §9).

---

## 1. Objectif et périmètre

### 1.1 Énoncé du sujet (rappel du cahier des charges)

| Phase | Contenu imposé |
|---|---|
| 1. Prétraitement | Rectification géométrique + alignement horizontal parfait du couple stéréo (gauche/droite) |
| 2. Extraction de caractéristiques | Correspondances de blocs de pixels via **StereoBM** ou **StereoSGBM** d'OpenCV, sur CPU |
| 3. Apprentissage | Génération de la carte de disparité + conversion mathématique par **triangulation** en carte de profondeur relative |
| 4. Reconnaissance / Inférence | Estimation des distances 3D des objets présents dans la scène |

### 1.2 Ce qu'on livre (definition of done)

Un dépôt Python exécutable en une commande qui :

1. télécharge et vérifie un jeu stéréo **avec vérité terrain** (disparité) ;
2. rectifie le couple d'images (démonstration complète par calibration, pas juste « on suppose ») ;
3. calcule une disparité dense (BM **et** SGBM) + post-traitements ;
4. convertit disparité → profondeur métrique → **nuage de points 3D** ;
5. mesure la distance réelle d'objets sélectionnés dans la scène ;
6. produit un **rapport de métriques** (taux de pixels erronés, RMSE, FPS) et les figures associées.

### 1.3 Hors périmètre (assumé)

- Pas de réseau de neurones profond (pas de MiDaS/RAFT-Stereo/CREStereo) : le sujet impose les méthodes classiques CPU. Le bonus ML précédemment envisagé est **écarté** (§9.1).
- Pas d'évaluation sur le *leaderboard* KITTI (dépôt de résultats = compte requis) : on reste en évaluation locale sur les vérités terrain publiques.

### 1.4 Définition de « pipeline classique » (terme employé dans ce plan)

« Classique » = **aucun modèle appris de bout en bout**. La disparité est calculée par corrélation de blocs (`StereoBM` / `StereoSGBM`), c'est-à-dire une fonction déterministe des pixels. Le seul « apprentissage » du pipeline porte sur des **paramètres** (calibration géométrique + grille de réglage), jamais sur des poids de réseau.

Concrètement, rester classique signifie :

| Inclus | Exclu |
|---|---|
| StereoBM / StereoSGBM / WLS, post-filtres OpenCV | réseaux de neurones (MiDaS, RAFT-Stereo, CREStereo, Depth Anything) |
| `stereoCalibrate` → `stereoRectify` → `remap` | tout pré-entraînement / tout GPU |
| grille d'hyperparamètres optimisée sur `tune` | classifieur ML même léger (option écartée, §9.1) |
| `reprojectImageTo3D` + export PLY | — |

Ce choix est cohérent avec le barème du cahier des charges (les 4 phases imposées sont 100 % classiques) et avec la contrainte matérielle (CPU-only, pas de GPU).

---

## 2. Données : réponse à la question « autre chose que Kaggle ? »

**Oui. Trois sources sans compte, sans clé API, sans formulaire — testées en direct (HTTP 206 sur requête partielle) :**

| Source | Accès | Poids | Intérêt |
|---|---|---|---|
| **Middlebury Stereo 2014** (principal, verrouillé) | URL directe par scène, `no auth` | ~58 Mo / scène (full res 2880×1988) | Vérité terrain dense **et** `calib.txt` (focale, baseline, doffs) → profondeur **métrique** |
| **Middlebury 2005 / 2003 / 2001** | ZIP directs | 5 Mo (tout 2005 en 1/3 rés.) / 1–3 Mo par scène | Mini-jeu de démarrage, format 16-bit PNG facile à lire |
| **KITTI Scene Flow / Stereo (zips S3)** | URL S3 directe, `no auth` | 1,68 Go / 2,01 Go | Scènes routières réalistes, baseline 0,54 m ; seul le *submission test* exige un compte |
| **`skimage.data.stereo_motorcycle()`** | Déjà local (scikit-image 0.24 installé) | 0 Mo | **Smoke test hors-ligne** : `left, right, disp` sans aucun téléchargement |

URLs vérifiées (utiles pour `scripts/fetch_data.py`) :

```
# Middlebury 2014 — une scène = 5 fichiers, tous en accès direct
https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-perfect/{im0.png,im1.png,disp0.pfm,disp1.pfm,calib.txt}
https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Backpack-perfect/...
# (23 scènes « perfect » : Adirondack, Backpack, Bicycle1, Cable, Classroom1, Couch, Flowers,
#  Jadeplant, Mask, Motorcycle, Piano, Pipes, Playroom, Playtable, Recycle, Shelves, Shopvac,
#  Sticks, Storage, Teddy, Vintage, ArtL, MotorcycleE, PianoL, PlaytableP)

# Middlebury 2005 — tout en une archive (1/3 de résolution, 2 vues)
https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/zip-2views/ALL-2views.zip   # 4,95 Mo

# Middlebury 2003 (Cones / Teddy)
https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/cones-png-2.zip

# Middlebury 2021 mobile — même schéma que 2014 (im0/im1/disp0.pfm/calib.txt)
https://vision.middlebury.edu/stereo/data/scenes2021/data/artroom1/im0.png

# KITTI — zips directs S3 (aucun login)
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip        # 1,68 Go (stereo+flow 2015)
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip      # 2,01 Go (stereo 2012)
```

### 2.1 Pièges d'accès identifiés (à coder dans le fetcher)

- **`User-Agent` obligatoire** : certains chemins Middlebury renvoient `403` à `urllib` sans UA navigateur. Le script enverra un UA explicite.
- **À éviter** : le miroir Hugging Face `SRPLB1/benchmarks-middlebury2014` est **gated** (`gated: manual`) → login requis. Les datasets HF publics ne demandent pas de token mais sont *rate-limités* : ce n'est pas notre voie principale.
- **ETH3D** : liens derrière un formulaire d'inscription → écarté (non testable sans compte).
- **Poids** : budget cible ≤ **700 Mo** → 8 scènes 2014 `-perfect` (~58 Mo chacune) + 4 `-imperfect` (~60 Mo chacune) + le zip 2005 (5 Mo). `disp*.pfm` pèse ~23 Mo/scène : on ne télécharge la GT que sur les scènes d'évaluation et pas sur celles de tuning.

### 2.2 Découpage train / val / test (pour ne pas tricher)

| Split | Scènes | Usage |
|---|---|---|
| `tune` | 6 scènes 2014 `-perfect` | grille de paramètres SGBM/BM |
| `val` | 4 scènes 2014 `-perfect` | sélection finale des paramètres, réglage des seuils |
| `test` | 4 scènes 2014 `-perfect` + 2 scènes 2021 `-perfect` | chiffres du rapport |
| `rect` | 4 scènes 2014 **`-imperfect`** | phase 1 : mesure de `dy` et re-rectification (jamais utilisées pour le réglage) |
| `demo` | `stereo_motorcycle` (skimage) + 2005 `ALL-2views.zip` | démo, figures, tests unitaires |

> **Note importante** : publier les chiffres sur les scènes de `tune` après avoir réglé les paramètres dessus serait une fuite méthodologique. Le rapport annoncera `val` **et** `test`.

---

## 3. Architecture du dépôt

```
sujet23_depth/
├── README.md                     # EN — pitch, install, 3 commandes de démo, résultats clés
├── PLAN_Sujet23_Depth_Estimation.md   # FR — ce document (document de travail)
├── docs/
│   ├── RAPPORT.md                # FR — rapport livrable (source Markdown → export PDF)
│   ├── RAPPORT.pdf               # FR — export final pour le rendu
│   └── figures/                  # FR — figures numérotées du rapport (générées par make report)
├── requirements.txt              # numpy, opencv-contrib-python, scikit-image, matplotlib, pytest
├── Makefile                      # make data / demo / eval / report / pdf / publish
├── .gitignore                    # data/, outputs/, .venv/, *.ply, docs/figures/*.png générés
├── scripts/
│   ├── fetch_data.py             # téléchargement + manifeste sha256 + reprise (aucun login)
│   └── make_report.py            # agrège les CSV de métriques → tableaux + figures du rapport
├── src/stereo_depth/
│   ├── config.py                 # dataclasses: DatasetCfg, MatcherCfg, DepthCfg (1 source de vérité)
│   ├── io_middlebury.py          # lecteur PFM, parseur calib.txt, chargeur générique "StereoPair"
│   ├── io_kitti.py               # lecteur calib KITTI + png 16-bit disparity (secondaire, M5+)
│   ├── calibrate.py              # damier: calibrateCamera / stereoCalibrate / RMS reprojection
│   ├── preprocess.py             # rectification, undistort/remap, downscale, gray, CLAHE, estimateur dy
│   ├── matcher.py                # StereoBM, StereoSGBM, WLS(contrib), post-filtres
│   ├── depth.py                  # disp→Z, reprojectImageTo3D, nuage de points, PLY
│   ├── objects.py                # sélection ROI/segment → distance objet (médiane robuste)
│   ├── evaluate.py               # BP-1.0, BP-2.0, RMSE/MAE, masque LRC, masque d'occlusion
│   └── cli.py                    # entrypoints: fetch / rectify / disparity / depth / eval / demo
├── tests/
│   ├── test_pfm.py               # PFM écrit/relu, y-flip, endianness
│   ├── test_depth_math.py        # Z = f·B/(d+doffs) vs valeurs analytiques
│   └── test_pipeline_smoke.py    # stereo_motorcycle → disparité non dégénérée
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # plages de disparité, ndisp, occlusions, dy avant/après
│   └── 02_results_analysis.ipynb      # courbes de réglage, cartes d'erreur, distances objets
├── demo/
│   └── live_demo.py              # UI OpenCV à sliders (démo de soutenance, cf. §10)
├── data/{raw,processed}/         # gitignored, + manifest.json
└── outputs/{figures,metrics,pointclouds}/  # gitignored (artefacts régénérables)
```

---

## 4. Conception technique par phase

### Phase 1 — Prétraitement & rectification (`preprocess.py`, `calibrate.py`)

**Constat vérifié** : les scènes Middlebury `-perfect` sont **déjà rectifiées** (`dyavg=0`, `dymax=0` dans `calib.txt`) et livrées avec leur géométrie. Charger ces images ne *démontre* donc pas la phase 1 : ça l'affirme. Mais le dataset contient aussi les variantes **`-imperfect`**, volontairement mal rectifiées — c'est le bac à sable idéal, sur données réelles, sans matériel :

| Scène `-imperfect` | `dyavg` | `dymax` | largeur | doffs |
|---|---|---|---|---|
| Adirondack | 0,735 px | 1,694 px | 2872 | 213,084 |
| Backpack | 0,649 px | 2,697 px | 2948 | 355,149 |
| Motorcycle | 0,918 px | 1,516 px | 2964 | 131,111 |
| Shelves | 0,447 px | 1,277 px | 2952 | 190,244 |

Un couple `-imperfect` est donc **réellement désaligné** (jusqu'à 2,7 px de décalage vertical résiduel), ce qui casse SGBM (la fenêtre de recherche suppose `dy=0`). La phase 1 devient mesurable : *avant* vs *après*.

Trois niveaux d'implémentation :

1. **Vérification** (obligatoire, toutes les scènes) : chargeur qui valide `im0/im1` même taille + cohérence `calib.txt` (`width`, `height`, `ndisp`), et **estimateur de `dy`** maison sur appariements ORB/SIFT (médiane + MAD du décalage vertical). Sortie : `dy_median` par scène dans le rapport. Sur `-perfect` on doit trouver ~0 (test de non-régression), sur `-imperfect` ~0,5–0,9 px.
2. **Re-rectification guidée par les données** (le cœur de la démo) : estimer la transformation corrective résiduelle (ECC affine `cv2.findTransformECC` / homographie estimée sur le plan dominant) sur une version 0,25×, puis `cv2.warpPerspective` sur la pleine résolution. Métriques **avant/après** : `dy_median`, `dymax`, et delta de BP-2.0 sur la même scène. C'est la figure qui prouve la phase 1, sans imprimante ni smartphone.
   - Contrôle de validité : la GT `disp0.pfm` d'une scène `-imperfect` est exprimée dans le repère *imperfect* → après re-rectification on doit **ré-échantillonner la GT** avec la même transformation (sinon les métriques sont fausses). Point d'implémentation à ne pas rater.
3. **Calibration damier réelle** (`calibrate.py`, optionnelle mais implémentée) : `cv2.calibrateCamera` + `cv2.stereoCalibrate` sur 15–25 paires de damier → `K, D, R, T` → `stereoRectify` → `initUndistortRectifyMap` → `remap`. Livre l'erreur de reprojection RMS et un couple « avant/après » sur tes propres photos. Si pas d'imprimante : damier affiché sur écran + capturé par un smartphone (distorsion faible mais réelle), ou **jeu synthétique à vérité connue** (rotation connue appliquée à l'image droite → la calibration doit retrouver exactement cette rotation : validation du code, pas d'un vrai objectif).

> **Réponse à « le dataset ne couvre-t-il pas déjà ça ? »** : il couvre la *géométrie* (focale, baseline, doffs) de ses propres scènes, mais il ne fournit **aucune paire non rectifiée** (`scenes2021/.../orig/` ne contient que la disparité et la description, vérifié). Donc : géométrie oui, *étape de rectification* non. Les variantes `-imperfect` sont le seul moyen de la démontrer sur données réelles — et c'est plus convaincant qu'un damier maison, car il y a une vérité terrain.

Réglages pratiques :
- Résolutions de travail : `1.0` (full, lent), `0.5` (défaut), `0.25` (rapide/CI). **Toute la géométrie (f, B, ndisp, doffs) est mise à l'échelle** — c'est la source n°1 de bugs, donc centralisée dans `config.py`.
- `cvtColor(BGR2GRAY)` + `clahe` optionnel (justifié : SGBM est sensible au contraste, l'égalisation aide sur les scènes sombres comme `Shelves`).
- `-imperfect` a des tailles d'image **différentes** de `-perfect` (2872 vs 2880, etc.) → aucune hypothèse de taille codée en dur.

### Phase 2 — Extraction de caractéristiques / matching (`matcher.py`)

- **StereoBM** : `numDisparities`, `blockSize=15`, `preFilterCap`, `textureThreshold`, `uniquenessRatio`.
- **StereoSGBM** : `minDisparity=0`, `numDisparities=288` (multiple de 16, dérivé de `ndisp=280` de `calib.txt`), `blockSize∈{5,7,9,11}`, `P1=8·c·bs²`, `P2=32·c·bs²` (réglage exponentiel via `λ = P2/P1`), `uniquenessRatio`, `speckleWindowSize`, `speckleRange`, `disp12MaxDiff=1` (contrôle gauche-droite intégré), `mode=STEREO_SGBM_MODE_SGBM_3WAY` puis `_HH`/`_SGBM` pour le compromis vitesse/qualité.
- **Post-filtres** : suppression des taches (speckle), `medianBlur(5)`, masque **LRC** (left-right consistency), et **WLS filter** (`cv2.ximgproc`) si `opencv-contrib-python` est installé (à ajouter au venv : le système n'a que `opencv-python` 5.0.0, **sans** `ximgproc`).
- Sortie : `disparity float32` en pixels **au facteur d'échelle de travail**, + masque `valid`.

### Phase 3 — « Apprentissage » = calibration + réglage statistique (`depth.py`, `config.py`)

Le cahier des charges appelle « apprentissage » la génération de la carte de disparité et la triangulation. On l'interprète rigoureusement comme **apprentissage des paramètres** (pas de réseau) :

1. **Paramètres géométriques** : soit lus dans `calib.txt` (`f=4161.221 px`, `B=176.252 mm`, `doffs=209.059`), soit **estimés** par `stereoCalibrate` (phase 1) — on compare les deux et on rapporte l'écart.
2. **Paramètres de matching** : recherche sur grille (`itertools.product` sur 4–5 hyperparamètres, ~200 combinaisons × 6 scènes à 0,25×) + affinage local sur les 10 meilleures. Objectif : minimiser BP-2.0 sur `tune`, sélection sur `val`.
3. **Modèle de profondeur** (triangulation) :

   ```
   Z = f · B / (d + doffs)          # convention officielle Middlebury (doffs = cx1 − cx0)
   X = (u − cx) · Z / f ;  Y = (v − cy) · Z / f
   ```
   Équivalent OpenCV : `Q` issue de `stereoRectify` → `cv2.reprojectImageTo3D(disp, Q)`. On implémente les **deux** et on vérifie qu'ils concordent à `<1e-6 près` (test unitaire). Unités : mm si `B` en mm (Middlebury) ; m si `B=0.54` (KITTI).
4. **Profondeur relative** (demandée par le sujet) : normalisation `Z/median(Z)` ou inversion `1/Z` pour l'affichage — la version métrique reste disponible pour l'inférence.

~~Option bonus — vrai modèle supervisé léger~~ → **écartée** (décision 4, §9.1). On garde le pipeline 100 % classique. La piste est documentée ici uniquement pour mémoire : un `RandomForestClassifier` (sklearn, CPU, <1 s) qui prédirait « pixel de disparité fiable / non fiable » à partir de descripteurs locaux (`|d_L − d_R|`, gradient local, résidu de coût, `uniquenessRatio`, taille de bloc) ferait gagner ~2 à 5 pts de BP-2.0, mais sort du périmètre du sujet. Le masque de confiance est donc obtenu par des moyens **déterministes** : LRC, seuils de texture, test de speckle.

### Phase 4 — Inférence 3D et distances (`objects.py`, `cli.py`)

- `stereo-depth --pair im0.png im1.png --calib calib.txt` → `disparity.png`, `depth.npy`, `depth_viz.png`.
- **Nuage de points** : `reprojectImageTo3D` → filtrage (`Z` fini, `Z < z_max`, masque de validité) → export **`.ply`** (`plyfile` ou writer maison) → visualisation Open3D *optionnelle* (dépendance lourde, on garde un PNG de projection + le `.ply` ouvrable ailleurs).
- **Distance d'objet** :
  1. sélection par ROI manuelle (`--roi x,y,w,h`) **ou** par segmentation grossière (plus grande composante connexe de la carte de profondeur seuillée / bbox de gradient) ;
  2. statistique **robuste** : médiane + MAD sur les pixels valides de la région (jamais la moyenne : les bords et les occlusions polluent) ;
  3. sortie : `{objet, distance_m, incertitude, %_pixels_valides}` en JSON + annotation sur l'image.
- **Mode vidéo** (optionnel) : paire stéréo webcam ou paire d'images par frame, résolution 0,25×, log FPS → démonstration « temps réel CPU ».

---

## 5. Évaluation — protocole et cibles

### 5.1 Métriques

| Métrique | Définition | Cible indicative |
|---|---|---|
| **BP-2.0** (bad pixel rate) | % pixels avec `\|d_est − d_gt\| > 2 px`, sur masque valide **hors occlusion** | ≤ 10 % (SGBM, 0,5×, Middlebury 2014) |
| BP-1.0 | idem avec seuil 1 px | ≤ 20 % |
| RMSE profondeur | en mm sur masque valide, scènes avec `calib` | ≤ 5 % d'erreur relative à 1 m |
| Erreur par bande | 0,5 / 1 / 2 m (Middlebury ≈ scènes de bureau) | décroissante, rapportée en table |
| Vitesse | FPS sur CPU (960×540) | ≥ 5 FPS (BM) / ≥ 2 FPS (SGBM-HH) |
| Mémoire | pic RSS | < 1,5 Go |

### 5.2 Masques (le point qui fait la différence dans la notation)

- `valid_mask = (disp_gt > vmin) & np.isfinite(disp_gt)`.
- **Occlusions** : `scenes2014` **ne fournit pas** de masque d'occlusion (vérifié : pas de `mask0nocc.png`). On le **dérive** par cohérence gauche-droite sur la vérité terrain : `|disp0[u] + disp1[u − disp0[u]]| > 1` → occlus. Méthode documentée et testée (comparée visuellement au masque d'occlusion officiel de 2005, disponible dans `scenes2005/test/occlusionMaps/`).
- Deux chiffres publiés : **avec** et **sans** occlusion (les deux sont standard dans la littérature).

### 5.3 Sorties du rapport

Tableaux CSV (`outputs/metrics/*.csv`), figures (paire stéréo, GT, BM, SGBM, carte d'erreur, profondeur colorisée, nuage de points), et **courbe de réglage** (BP-2.0 vs `blockSize`/`λ`/`numDisparities`) — c'est ce qui montre que la phase 3 a réellement eu lieu.

---

## 6. Jalons (milestones)

| # | Jalon | Livrable | Done quand |
|---|---|---|---|
| **M0** | Squelette + données + dépôt | `requirements.txt`, venv, `fetch_data.py`, `stereo_motorcycle` chargé, dépôt GitHub public | `make data` télécharge les scènes et passe le manifeste sha256 ; `git push` OK |
| **M1** | Lecture & rectification | `io_middlebury.py`, `preprocess.py`, `calibrate.py`, notebook 01 | `dy_median ≈ 0` sur les `-perfect`, `dy ≈ 0,45–0,92 px` sur les `-imperfect`, **puis < 0,15 px après re-rectification** (test chiffré) |
| **M2** | Matching | `matcher.py`, CLI `disparity` | BM et SGBM produisent une disparité non dégénérée sur `motorcycle` et 3 scènes 2014 |
| **M3** | Profondeur & 3D | `depth.py`, export PLY | `Z = f·B/(d+doffs)` ≡ `reprojectImageTo3D` (test unitaire), PLY ouvrable |
| **M4** | Réglage statistique | `tune` + CSV de grille | meilleurs params figés dans `config.py`, sélection sur `val` |
| **M5** | Inférence objets + éval (+ KITTI) | `objects.py`, `evaluate.py`, `io_kitti.py`, CLI `eval` | métriques sur `test` + distances d'objets en JSON + au moins une scène KITTI passée dans le même pipeline |
| **M6** | Démo live + rapport | `demo/live_demo.py`, notebook 02, `docs/RAPPORT.md` → `.pdf` | `make demo-live` tourne à ≥ 5 FPS à 0,5×, PDF généré, `make test` vert, README (EN) à jour |

Ordre de travail conseillé : M0 → M1 → M2 (chemin critique) puis M3/M4 en parallèle, M5 conditionné par M2.

---

## 7. Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Format **PFM** (float32, ligne d'en-tête, y-flip) mal lu | GT fausse → métriques fantaisistes | lecteur dédié + test aller-retour + contrôle `vmin/vmax` du `calib.txt` vs `min/max` de la GT |
| **`doffs` ignoré** | biais constant sur toute la profondeur (≈ +139 % d'erreur sur `Adirondack`) | paramètre obligatoire dans `Calib`, défaut lu dans `calib.txt`, test analytique. **Formule officielle : `Z = B·f/(d + doffs)`** (vérifiée sur la page de description du dataset, 9 juil. 2015) — un signe `−` a été corrigé après l'implémentation de M0 |
| Mise à l'échelle (0,25×/0,5×) appliquée à `f` mais pas à `ndisp` | disparité tronquée | une seule fonction `scaled_calib(scale)` |
| Images 2880×1988 → SGBM très lent / RAM | itération pénible | travail à 0,5× par défaut, 0,25× en CI, full res réservé à la figure finale |
| Régions sans texture (murs, ciel) | trous de disparité | masque `textureThreshold`, interpolation, et surtout **ne pas** compter ces pixels comme échec (masque valide) |
| `opencv-contrib` absent (système = opencv-python 5.0.0) | pas de WLS | WLS marqué *optionnel* avec dégradation propre (`try/except ImportError`) |
| Sur-réglage sur `tune` | chiffres non crédibles | split strict + rapport de `val` et `test` séparés |
| KITTI 1,7 Go lourd / licences | temps perdu | **optionnel** : branche `io_kitti.py` seulement si M5 est bouclé |

---

## 8. Aide-mémoire (commandes prévues)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv-contrib-python, scikit-image, matplotlib, pytest

make data       # 8 scènes Middlebury 2014 (-perfect + -imperfect) + vérif sha256   (~700 Mo, sans login)
make demo       # stereo_motorcycle → disparité + profondeur + PLY (hors-ligne)
make eval       # BP-1.0 / BP-2.0 / RMSE sur le split test
make report     # figures + CSV agrégés dans outputs/ et docs/figures/
make pdf        # docs/RAPPORT.md → docs/RAPPORT.pdf (pandoc + tectonic, français)
make publish    # git push vers le dépôt GitHub public

# usage unitaire
python -m stereo_depth.cli disparity --left data/raw/scenes2014/Adirondack-perfect/im0.png \
                                     --right .../im1.png --calib .../calib.txt --method sgbm --scale 0.5
python -m stereo_depth.cli depth --disp outputs/disparity.npy --calib .../calib.txt --ply out.ply
python -m stereo_depth.cli rectify --scene Adirondack-imperfect --report-csv outputs/metrics/dy.csv
```

Environnement constaté :

| Élément | Constaté |
|---|---|
| Machine | **Apple M4**, 10 cœurs, 16 Go, macOS 27.2.1 (arm64) → largement suffisant pour SGBM temps réel à 0,5× |
| Python système | 3.9.6 (`/usr/bin/python3`), `numpy 2.0.2`, `scikit-image 0.24.0` |
| OpenCV système | `opencv-python 5.0.0` — **sans** `ximgproc` (pas de WLS) → venv avec `opencv-contrib-python` |
| Chaîne rapport | `pandoc` 3.x, `tectonic` 0.17, BasicTeX (`xelatex`), `weasyprint` — tous présents |
| Git / GitHub | `gh` authentifié (`fifmoh`, protocole https), `git` configuré |
| Disque | ~119 Go libres → aucun souci de cache datasets |

> Note PDF : Pandoc + **tectonic** plutôt que `xelatex` nu, car tectonic télécharge automatiquement les paquets LaTeX manquants (BasicTeX est minimal et n'a pas forcément `babel-french` / `fontspec`).

---

## 9. Décisions tranchées et questions encore ouvertes

### 9.1 Tranché

| # | Question | Décision |
|---|---|---|
| 1 | Dataset principal | **Middlebury Stereo 2014** ; KITTI en branche secondaire optionnelle après M5 |
| 2 | Langue | **README/code : EN** — **rapport : FR** (§ « Décisions verrouillées ») |
| 3 | Phase 1 / calibration | **Oui, implémentée** ; démonstration principale sur les scènes **`-imperfect`** (dy réel 0,45–0,92 px), damier en complément (§4 phase 1) |
| 4 | Bonus RandomForest de fiabilité | **Écarté** — pipeline 100 % classique (§1.4) |
| 5 | Livrable de démo | **CLI + notebooks + UI OpenCV à sliders** (cf. §10), périmètre complet maintenu |
| 6 | Format du rapport | **Markdown → PDF via Pandoc + tectonic**, en français, dans `docs/` |
| 7 | KITTI / UI slider | **Dans le scope** (conservés, pas de coupe) |
| 8 | Rendu | **Dépôt GitHub public** (`gh` déjà authentifié) ; notebooks inclus dans le rendu |
| 9 | Soutenance | **Démo live requise** → spec §10 |

### 9.2 Dernières questions (non bloquantes)

1. **Nom du dépôt GitHub public** : proposition `stereo-depth-3d-vision` (alternative : `sujet23-depth-estimation`). À valider avant le `create`.
2. **Échéance** : date de rendu ? (non bloquant, mais conditionne l'ordre M4 → M5)
3. **Imprimante / smartphone** pour le damier : si oui, on ajoute une figure « calibration réelle » (~1 h) ; si non, la variante synthétique + `-imperfect` suffisent largement.

---

## 10. Démo live (soutenance)

**Format** : `make demo-live` → une seule fenêtre OpenCV, 3 panneaux côte à côte, sur une scène chargée en 0,5× :

| Panneau | Contenu |
|---|---|
| Gauche | image gauche brute |
| Milieu | **carte de disparité** colorisée (`COLORMAP_TURBO`), mise à jour à chaque mouvement de slider |
| Droite | **carte de profondeur métrique** + distance au point survolé/cliqué |

**Contrôles** :
- trackbars : `blockSize` (5–21, impairs), `numDisparities` (16–320, pas 16), `uniquenessRatio` (0–20), `speckleWindow`, `λ = P2/P1`, toggle WLS.
- touches : `b`/`s` = BM ou SGBM, `n`/`p` = scène suivante/précédente, `g` = GT superposée, `d` = afficher `dy` avant/après rectification, `q` = quitter.
- clic gauche sur l'image → affiche `Z = … mm à ce pixel` + trace du rayon dans le nuage de points.
- overlay permanent : **FPS** et nom de la scène.

**Scénario de démo (3–4 min)** :
1. `stereo_motorcycle` (hors-ligne) → pipeline complet en 2 s, PLY exporté.
2. `Adirondack-imperfect` → `d` : montrer `dy≈0,74 px` → re-rectifier → `dy≈0,1 px` (fig. phase 1).
3. `Adirondack-perfect` → bouger `blockSize` et `numDisparities` en direct → montrer l'effet sur le bruit et les trous + BP-2.0 affiché en overlay (fig. phase 2/3).
4. Clic sur un objet → distance en mm, comparée à la GT (`calib.txt`) → fig. phase 4.
5. Plan B si la machine du jury rame : `02_results_analysis.ipynb` avec les figures déjà générées (aucune dépendance à l'inférence live).

**Pourquoi la machine compte** : les cibles FPS (≥ 5 à 0,5×) et la fluidité des sliders dépendent du CPU ; l'Apple M4 (10 cœurs) testera confortablement SGBM en 960×540, mais si la démo tourne sur un portable du lycée/campus, il faut prévoir un `--scale 0.25` par défaut. Les deux chemins sont testés en M6.

---

## 11. Prochaine action (M0)

1. `git init`, `.gitignore`, `requirements.txt`, venv, `README.md` (EN) squelette
2. `scripts/fetch_data.py` (UA navigateur, manifeste sha256, reprise) + téléchargement réel de 3 scènes `-perfect` et 2 `-imperfect`
3. smoke test `skimage.data.stereo_motorcycle()` → disparité non dégénérée (prouve que la toolchain tourne avant tout téléchargement)
4. `gh repo create stereo-depth-3d-vision --public` + premier commit poussé

Je m'arrête là et j'attends le feu vert (et le nom du dépôt) avant de créer le dépôt public.

---

## 12. Journal d'exécution — M0 livré (vérifié sur données réelles)

### 12.1 Ce qui tourne

| Élément | État |
|---|---|
| venv + `requirements.txt` | ✅ `opencv-contrib-python 5.0.0` (avec `ximgproc`), `numpy 2.0.2`, `skimage 0.24`, `pytest 8.4` |
| `scripts/fetch_data.py` | ✅ 26 fichiers / **299 Mo** téléchargés sans aucun login, manifeste sha256 |
| Scènes locales | `Adirondack`, `Backpack`, `Bicycle1` (-perfect) + `Adirondack`, `Backpack` (-imperfect) + archive 2005 |
| `io_middlebury.py` | ✅ lecteur PFM (+writer), parseur `calib.txt`, `Calib.scaled()` |
| `preprocess.py` | ✅ estimateur `dy` **sub-pixel** (ORB + `cornerSubPix` + affine robuste) |
| `matcher.py` | ✅ StereoBM / StereoSGBM, LRC, suppression de taches |
| `depth.py` | ✅ `Z = f·B/(d + doffs)`, `reprojectImageTo3D`, PLY-ready |
| `cli.py` | ✅ `info`, `dy`, `disparity` (sorties JSON) |
| Tests | ✅ **26 tests verts** (PFM, calib, maths profondeur, rectification, smoke offline) |

### 12.2 Deux corrections de fond apportées au plan

1. **Formule : `Z = f·B/(d + doffs)`** et non `(d − doffs)`. Vérifié sur la page de description
   officielle du dataset (`Z = baseline * f / (d + doffs)`, `doffs = cx1 − cx0`). L'erreur
   de signe a été détectée par les tests unitaires, puis corrigée partout.
2. **L'estimateur `dy` doit être sub-pixel + affine.** Un simple médian sur des points ORB
   entiers renvoie `0.0` même sur les scènes `-imperfect` (quantification). Il faut
   `cornerSubPix` + un modèle affine `dy(x,y) = a·x + b·y + c` pour retrouver les valeurs
   publiées.

### 12.3 Résultats mesurés (phase 1, en pixels pleine résolution)

| Scène | `dyavg` publié | `dymax` publié | ⚠️ `dyavg` mesuré | `dymax` mesuré |
|---|---|---|---|---|
| `Adirondack-perfect` | 0,000 | 0,000 | **0,057** | 0,156 |
| `Adirondack-imperfect` | 0,735 | 1,694 | **0,550** | 1,503 |
| `Backpack-imperfect` | 0,649 | 2,697 | 0,081 | 0,157 |

Contrôle profondeur (`Adirondack-perfect`, SGBM 0,5×) : 76 % de pixels valides,
profondeur médiane **2,21 m** (p10–p90 : 1,91–2,87 m) → physiquement cohérent.

### 12.4 Conséquence pour M1 (à retenir)

Le `dy` seul **ne suffit pas** comme critère de phase 1 : sur `Backpack-imperfect` (textures
répétitives) l'estimateur ne voit que 0,08 px au lieu de 0,65 px. M1 doit donc :

1. mesurer `dy` **et** le gain de qualité de disparité (BP-2.0 avant / après re-rectification) ;
2. valider la re-rectification sur des corruptions **à réponse connue** (rotation 0,3°,
   translation 4 px) — déjà testé et fonctionnel ;
3. garder la calibration damier (`calibrate.py`) comme démonstration de `K, D, R, T`,
   pas comme unique preuve.
