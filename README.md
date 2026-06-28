# Détection de fraude sur des transactions retail

Petit projet de data / machine learning où on essaie de repérer les transactions frauduleuses dans un jeu de paiements en ligne. L'idée de départ : à partir d'un montant, d'un appareil, d'une localisation et de quelques signaux d'alerte, est-ce qu'on peut deviner si une transaction est une fraude ou non ?

On bosse sur ~100 000 transactions avec une vingtaine de colonnes. La cible c'est `fraud_flag` (0 = transaction normale, 1 = fraude).

## Ce qu'il y a dans le projet

On a procédé en deux temps. D'abord comprendre les données (qu'est-ce qui ressort, quelles colonnes ont l'air liées à la fraude), puis construire un modèle pour prédire.

Les fichiers de code :

- **`codeV1`** — toute la partie exploration. Distribution de la cible, matrice de corrélation, histogrammes, taux de fraude selon le type d'appareil. C'est là qu'on a creusé pour comprendre le dataset avant de modéliser.
- **`graphe_binaire`** — une visualisation qu'on aimait bien : on découpe le plan en 4 quadrants en croisant deux signaux (transactions rapprochées + localisation inhabituelle) et on regarde le taux de fraude dans chaque zone. C'est assez parlant.
- **`Code_main`** — un nuage de points où la couleur indique la fraude, la forme l'appareil à risque et la taille le niveau de risque. Beaucoup d'infos d'un coup.
- **`MLfinal`** — le modèle final. Un peu de feature engineering, un score "maison" (`prob_fraud`), une régression logistique et une courbe ROC pour voir ce que ça donne.
- **`formule_ICC`** — la formule du score de risque qu'on a bricolée.

Et les données + les images : `retail_fraud_detection_100k.csv` pour le dataset, et les `.png` pour les graphes exportés.

## Les données

Le CSV contient pas mal d'infos sur chaque transaction : le montant, la méthode de paiement (carte, PayPal, Apple Pay...), le type d'appareil, si c'est une transaction internationale, le nombre d'échecs sur 24h, l'âge du compte, etc. Et plusieurs "flags" qui sont des signaux d'alerte (montant inhabituel, localisation bizarre, appareil à risque...).

Un truc pratique : le dataset est plutôt équilibré, environ la moitié des transactions sont frauduleuses. Ça change des cas classiques de détection de fraude où on a 0,1 % de fraude et où tout est galère à entraîner.

## Comment on s'y est pris

Pour le modèle on a d'abord créé deux variables qui nous semblaient utiles :
- le rapport (en log) entre le montant de la transaction et le montant moyen des 7 derniers jours — pour repérer quand quelqu'un dépense d'un coup beaucoup plus que d'habitude
- le taux d'échec des transactions sur 24h

Ensuite on combine tout ça dans un score `prob_fraud`, et on entraîne une régression logistique dessus (avec `class_weight="balanced"`). On évalue avec l'AUC de la courbe ROC.

## Ce qu'on a trouvé

Le résultat le plus marquant vient de la visualisation par quadrants : plus on accumule de signaux d'alerte, plus le risque grimpe. Quand aucun signal n'est levé, on est autour de 11 % de fraude. Mais quand transactions rapprochées et localisation inhabituelle se combinent, ça monte à ~88 %. Bref, c'est vraiment la combinaison des signaux qui compte, pas un signal isolé.

## Pour faire tourner le code

Il faut ces librairies :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Ensuite pour l'exploration :

```bash
python codeV1
```

Et pour le modèle + la courbe ROC :

```bash
python MLfinal
```

La courbe ROC se sauvegarde dans `courbe_roc_finale.png`.

## Stack

Python, pandas, numpy, matplotlib, seaborn, scikit-learn. Rien d'exotique.
