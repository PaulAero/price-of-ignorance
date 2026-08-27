## Verdict

Le bon diagnostic n’est pas « trop simple ». C’est **(c) simpliste au mauvais endroit**.

Le jeu, l’API en une fonction et les 155 lignes sont au bon niveau pour la cible. En revanche, la rigueur expérimentale, le tracker présenté comme énigme et les affirmations de nouveauté sont actuellement trop faibles. Dans cet état, je ne retweeterais pas le dépôt : un lecteur technique attentif peut casser le hook ou la crédibilité en quelques minutes.

## 1. Audit technique

Exécution :

- `python ignorance.py` reproduit exactement le README.
- `python test_ignorance.py` affiche `all tests pass`.
- `pytest` n’est pas installé, ce qui est cohérent avec « zéro dépendance » ; le lanceur direct suffit.

### Critique — l’action épistémique modifie le monde

Un unique générateur alimente à la fois :

- la dynamique d’usure ;
- les nouveaux taux d’usure ;
- le bruit du capteur.

Voir [ignorance.py:38](../ignorance.py:38), [ignorance.py:58](../ignorance.py:58) et [ignorance.py:67](../ignorance.py:67).

Un `check` consomme donc des nombres aléatoires et change les futures trajectoires physiques. J’ai comparé deux politiques qui ne font aucune intervention physique, dont l’une effectue seulement un check initial :

| Politique | doing | failing | total hors/coût info |
|---|---:|---:|---:|
| toujours `run` | 96.775 | 967.750 | 1064.525 |
| `check` à t=0, puis `run` | 96.000 | 960.000 | 1056.000 |

Une observation censée être purement épistémique change donc les pannes. La phrase « same machine, same seeds » est fausse au sens causal et l’appariement des scores est compromis. Il faut deux flux RNG indépendants, idéalement une trajectoire de dynamique pré-générée.

### Critique — le tracker répare fréquemment deux fois

Après une lecture supérieure au seuil, le tracker retourne `fix` sans remettre son état interne à zéro : [ignorance.py:111](../ignorance.py:111). À l’appel suivant, l’estimation reste élevée et déclenche souvent un deuxième `fix`.

Sur 2 000 épisodes :

- 7 902 actions `fix` ;
- 2 596 réparations immédiatement consécutives ;
- score livré : 408.6 ;
- avec seulement ce reset corrigé : 360.6.

Le tracker corrigé perd encore largement contre 255.1 : le phénomène de modèle moyen aveugle aux machines rapides est donc réel. Mais environ 48 points de sa défaite viennent d’un bug. Le README attribue toute la défaite aux « bad batches » : c’est trompeur.

### Important — sémantique temporelle ambiguë

Lors d’une panne, `_renew()` est appelé avant de produire la lecture. Un `check` au pas de la panne retourne donc une mesure du composant neuf, accompagnée de `failed=True`, pas une mesure du composant qui vient de casser. Voir [ignorance.py:60](../ignorance.py:60).

Autre ambiguïté : « breakdown costs $500 » signifie dans le code $500 de failure **plus** $50 de remplacement forcé. Le coût réel immédiat est donc $550.

Ce n’est pas nécessairement un mauvais modèle, mais l’ordre `action → usure → panne/remplacement → observation` doit être explicite et testé.

### Tests trop faibles

Le commentaire annonce « scoreboard reproducibility », mais aucun test ne vérifie les valeurs du scoreboard : [test_ignorance.py:1](../test_ignorance.py:1).

Il manque notamment :

- l’invariance physique entre `run` et un `check` dont la lecture est ignorée ;
- une régression contre les doubles réparations ;
- l’ordre exact check/panne/remplacement ;
- `reward == -sum(costs)` ;
- les bornes de l’horizon et les actions invalides ;
- des résultats par épisode permettant des comparaisons appariées.

## 2. Est-ce trop simple pour Twitter/CS majors ?

Non : **la simplicité est bien le produit**. Installation nulle, exécution immédiate, trois coûts compréhensibles et une politique modifiable en quelques lignes sont exactement les bons choix.

Mais les petits dépôts reconnus comme élégants ont généralement quatre propriétés :

1. Ils compriment un mécanisme réel et canonique, pas seulement un toy arbitraire.
2. Ils accomplissent quelque chose d’étonnamment complet.
3. Leur exactitude est démontrée par un oracle ou un résultat visible.
4. Chaque simplification conserve l’invariant central.

`micrograd` ne fait pas seulement 150 lignes : son moteur entraîne réellement un MLP, possède des visualisations et compare ses gradients à PyTorch dans ses tests. [micrograd](https://github.com/karpathy/micrograd)  
`minGPT` est une implémentation réelle de GPT, avec tâches de démonstration et notebooks, pas une métaphore de Transformer. [minGPT](https://github.com/karpathy/minGPT/)  
Même tinygrad précise que le but n’est pas le code golf, mais la réduction de complexité avec lisibilité, tests et benchmarks. [tinygrad](https://github.com/tinygrad/tinygrad)

Ce dépôt possède la lisibilité et l’exécution immédiate. Il lui manque encore :

- un invariant causal correct ;
- un oracle ou une référence crédible ;
- un résultat visuel mémorable ;
- une boucle de contribution effectivement opérationnelle.

Il est donc proche d’une bonne miniature, mais pas encore d’un « micrograd du coût de l’information ».

## 3. Le hook et la profondeur

Le classement annoncé est réel sur le code livré :

- périodique : 255.1 ;
- tracker : 408.6.

Mais le hook mélange actuellement deux histoires :

- histoire intéressante : un modèle moyen gère mal l’hétérogénéité persistante ;
- histoire embarrassante : le tracker paie des réparations doubles à cause de son état.

Il faut corriger le tracker puis annoncer honnêtement le résultat restant, environ 360 sur le simulateur actuel. Le hook sera moins spectaculaire, mais beaucoup plus solide.

### Marge de progression

La cible 255.1 n’est pas tunée :

- `check_then_fix(k=4, threshold=0.40)` donne 253.38 ;
- cependant l’amélioration de 1.75 est inférieure au bruit : SE appariée ≈ 4.21 ;
- l’IC 95 % individuel de 255.1 est environ ±6.6.

« Beat 255.1 » permet donc des victoires dues au bruit ou à une modification triviale de paramètres.

J’ai écrit une règle légale très simple :

- check tous les 4 pas ;
- après une lecture ≥ 0.25, repasser à tous les 2 pas ;
- réparer à 0.50.

Résultats :

- 223.1 sur les graines publiques ;
- 225.1 sur 2 000 autres graines ;
- environ 227 sur un lot de 10 000 graines selon la variante.

La marge communautaire est donc réelle : environ 30 points sont récupérables sans RL ni gros code.

Un oracle illégal disposant gratuitement de l’usure réelle, avec un seuil proche de 0.75, obtient environ 121. Ce n’est pas une preuve de l’optimum, mais un plancher optimiste utile. Je présenterais donc :

- ~225 : meilleure référence légale trouvée rapidement ;
- ~121 : référence oracle, explicitement non admissible ;
- optimum légal : inconnu, probablement assez loin de 255 pour soutenir un challenge.

La profondeur existe. Elle n’est simplement ni mesurée ni mise en scène.

## 4. Priorités

### P0 — avant toute promotion

- Séparer RNG physique et RNG capteur ; ajouter le test de non-intervention épistémique.
- Corriger le tracker, l’ordre temporel et recalculer tous les scores.
- Remplacer les affirmations absolues de nouveauté. Les observations coûteuses et actions de mesure existent déjà en active sensing, AMRL et ACNO-MDP : par exemple [NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/83e8fe6279ad25f15b23c6298c6a3584-Abstract.html) et [Active Measure RL](https://publications-cnrc.canada.ca/eng/view/object/?id=0a738f55-7c86-4259-9a0a-a1a1e882e8c8). La revendication défendable est un **contrat d’API standardisé avec canal de coût séparé**, pas l’invention de l’observation coûteuse.
- Construire le leaderboard promis : registre de soumissions, résultats par épisode, génération automatique du tableau, CI réelle et critère d’acceptation statistique.
- Utiliser un jeu public de développement et une adjudication held-out. Les graines publiques 0–1999 sont directement sur-ajustables.

### P1 — pour rendre le dépôt partageable

- Une image partageable : deux trajectoires, usure cachée, checks, fixes et panne, puis les trois barres doing/knowing/failing.
- README en trois temps : problème en 15 secondes, surprise, bouton/challenge.
- Fournir un `submission_template.py` minimal et une commande unique d’évaluation.
- Ajouter une solution forte dans un `SPOILERS.md`, ou la révéler après une première manche ; ne pas prétendre qu’elle est optimale.
- Ajouter un mode difficile facultatif : capteur biaisé, délai d’inspection ou mélange explicite de lots rapides/lents.
- Colab ou exécution navigateur : pas techniquement nécessaire, mais utile pour convertir les curieux non-cloneurs.

### P2

- Social preview, badges CI et compteur de soumissions.
- Template de PR avec score public, score held-out et ventilation des coûts.
- Petit historique des records avec nom, lien vers la politique et taille en lignes.
- Lien vers l’essai lorsqu’il existe ; « essay: coming soon » donne aujourd’hui une impression d’inachevé.

## 5. Retweet et risque réputationnel

Je retweeterais une version où :

- le `check` ne change réellement que l’information ;
- le tracker perd pour une raison intellectuellement honnête ;
- une politique communautaire peut passer de 255 à ~225 ;
- une CI held-out confirme le record ;
- un visuel rend la surprise compréhensible sans ouvrir le code ;
- la nouveauté est formulée sans nier une littérature existante.

Le risque de desservir `coin-envs` est **réel et actuellement moyen à élevé**. Pas parce que le toy est simple, mais parce qu’il se présente comme la miniature d’un travail « sérieux, audité » tout en ayant un invariant causal cassé, une baseline boguée, une CI inexistante malgré la promesse, et une revendication historique réfutable.

Corriger ces points transformerait la simplicité en preuve de maîtrise. Dans l’état actuel, elle peut être lue comme absence de rigueur.

## Note

**5/10 pour l’adéquation à l’objectif déclaré.**  
Le concept et le temps-to-understanding sont bons ; l’intégrité expérimentale et la boucle virale ne sont pas encore au niveau nécessaire.
