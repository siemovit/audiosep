# MVA 2025 : Cours Deep et signal, liste des mini projets

Le mini-projet est à faire en binômes.

Le mini-projet amène à un notebook jupyter structuré que vous m'enverrez en
présenterai lors d'une soutenance de 15 minutes (prévoir 12 minutes de parole)
en visiocoférence. Vous pouvez présenter directement votre notebook et/ou faire
des slides si vous le souhaitez mais ce n'est pas exigé.

3 journées pour les soutenances :

- le mardi 16 décembre 2025
- le mercredi 17 décembre 2025
- le vendredi 09 janvier 2026 Merci de vous inscrire en binôme sur un créneau
  ici : https://framadate.org/ctziX5f4fTo4l2Xn (! merci de ne prendre qu'un seul
  créneau)

Voici une liste de propositions de mini-projets. Vous pouvez proposer votre
sujet indépendamment de cette liste ou proposer une variante d'un des sujets
proposés. Dans ce cas validez vos propositions avec moi. Il faut que le sujet
reste dans les thématiques vues en cours.

L'objectif est de travailler environ 10h sur le projet (par personne donc 20h
par binôme). Pour chaque sujet vous êtes invités à prendre des initiatives
notamment pour

- analyser les données (statistiques haut niveau, visualisation, évaluation des
  difficultés)
- partir du cours ou d'un article lié au sujet traité que vous aurez identifié
  dans une (rapide) étude bibliographique
- définir une ou plusieurs métriques d'évaluation
- définir et implémenter une méthode baseline dont les performances vous
  serviront de référence
- implémenter au moins une et idéalement deux méthodes traitant le problème
  considéré. Au moins une des deux méthodes est une approche par apprentissage
  profond. La seconde peut être une approche de traitement du signal classique,
  une autre architecture de réseau, la même architecture avec une stratégie
  d'augmentation de données ou d’ingénierie des données...
- comparer les approches quantitativement et analyser qualitativement vos
  résultats, les cas de succès et les cas d'échecs.

Si vous n'avez pas abouti à des résultats probants sur cette durée vous êtes
invités à analyser de façon critique vos résultats et émettre des hypothèse sur
ce qui n'a pas fonctionné (type d'approche, architecture, qualité des
données...). Une bonne analyse de vos résultats et une méthodologie rigoureuses
seront largement valorisées dans l'évaluation, vos initiatives aussi.

Pour chaque sujet, vous êtes libres d'explorer les pistes qui vous intéressent
(tant qu'elles sont raisonnables). Si vous avez des idées originales n'hésitez
pas à être créatifs !

## Audio

### Denoising Pour ce projet vous avez :

Pour le train

- Un dossier contenant des fichiers d'enregistrements de voix sans bruit
  (audio/voice_origin/train)
- Un dossier contenant des fichiers d'enregistrements de voix avec une ambiance
  de rue en arrière-plan (audio/denoising/train) La correspondance entre un
  enregistrement avec ambiance et l'enregistrement parfait de la voix se fait
  via le nom des fichiers.

Pour l'ensemble de test vous avez deux ensembles de fichiers similaires.

Dans audio/voice_origin et audio/denoising vous avec un dossier train_small de
petite taille que vous pouvez télécharger rapidement pour faire des essai.

L'objectif est d'estimer à partir du signal bruité le signal de voix.

Les signaux ont un SNR (Signal to Noise Ratio) compris entre 0 et 20 dB.

Vous pouvez au choix travailler

- sur le spectrogramme par exemple en vous des approches par masquage présentés
  dans le cours 09 et en estimant les masques avec un réseau Seq2Seq de votre
  choix ou un UNet (cf A. Jansson et Al., SINGING VOICE SEPARATION WITH DEEP
  U-NET CONVOLUTIONAL NETWORK, ISMIR 2017 )
- directement sur la forme d'onde :
  - cf D. Stoller et Al., WAVE-U-NET: A MULTI-SCALE NEURAL NETWORK FOR
    END-TO-END AUDIO SOURCE SEPARATION, ISMIR 2018
  - les apporches TAS NEt : Y. Luo et Al., TaSNet: Time-Domain Audio Separation
    Network for Real-Time, Single-Channel Speech Separation, ICASSP 2018 ou Y.
    Luo et Al., Conv-tasnet: surpassing ideal time–frequency magnitude masking
    for speech separation. IEEE/ACM Transactions on Audio, Speech, and Language
    Processing, 2019.

Libre à vous de choisir la fonction de perte utilisée dans l’entraînement et
adaptée au format des données que vous utiliserez en entrée du réseau de
neurones.

Pour l'évaluation des performances sur l'ensemble de test, outre la fonction de
perte vous vous intéresserez au PESQ et au STOI des voix estimées.

## Séparation de sources

L'objectif de ce projet est d'estimer conjointement la composante voix et la
composante bruit d'un enregistrement audio. Pour ce projet vous avez :

Pour le train

- Un dossier contenant des sous dossier numérotés (exemple 0001 ou 1256)
- Dans chaque sous dossier vous avez trois fichiers wav : mix_snr_XX.wav ,
  voice.wav, noise.wav
- voice.wav et noise.wav sont les vérités terrain à estimer, mix_snr_XX.wav est
  le mélange des deux sources avec un SNR de XX pour la composante voix (et de
  -XX pour la composante bruit)

L'ensemble de test est constitué de la même façon.

Vous pouvez au choix travailler

- sur le spectrogramme par exemple en vous des approches par masquage présentés
  dans le cours et en estimant les masques avec un réseau Seq2Seq de votre choix
  ou un UNet (cf A. Jansson et Al., SINGING VOICE SEPARATION WITH DEEP U-NET
  CONVOLUTIONAL NETWORK, ISMIR 2017 )
- avec la méthode Deep Clustering : J.R. Hershey et Al., Deep clustering:
  Discriminative embeddings for segmentation and separation, ICASSP 2016
- directement sur la forme d'onde :
  - cf D. Stoller et Al., WAVE-U-NET: A MULTI-SCALE NEURAL NETWORK FOR
    END-TO-END AUDIO SOURCE SEPARATION, ISMIR 2018
  - les apporches TAS NEt : Y. Luo et Al., TaSNet: Time-Domain Audio Separation
    Network for Real-Time, Single-Channel Speech Separation, ICASSP 2018 ou Y.
    Luo et Al., Conv-tasnet: surpassing ideal time-frequency magnitude masking
    for speech separation. IEEE/ACM Transactions on Audio, Speech, and Language
    Processing, 2019.

Libre à vous de choisir la fonction de perte utilisée dans l’entraînement et
adaptée au format des données que vous utiliserez en entrée du réseau de
neurones.

### Complétion de paquets perdus

Lors d'une communication sur IP les échantillons sont envoyés sous forme de
paquets. Un paquet a une certaine probabilité d'être perdu ce qui produit un
"trou" dans le signal reçu. L'objectif de ce projet est de combler les trous
dans un signal de voix.

Le dataset de train se compose de :

- Un dossier contenant des fichiers d'enregistrements de voix à 8kHz
  (audio/voice_origin/train) ; ce sont les données objectifs
- Un dossier contenant les fichiers correspondants avec certains paquets qui ont
  été perdu (audio/packet_loss/train) ; ce sont les données d'entrée à traiter

La correspondance entre les deux dossiers pour reconstituer les paires (donnée
d'entrée, vérité terrain) se fait par le nom de fichier. Les données de test
sont structurées de la même façon.

Vous pouvez au choix travailler sur spectrogramme ou directement sur la forme
d'onde en vous inspirant par exemple des références données pour la séparation
de sources (UNet, WavUnet)

Pour ce sujet vous implémenterez nécessairement une approche classique par
exemple en utilisant des interpolations et une méthode à base de réseaux de
neurones profonds.

Libre à vous de choisir la fonction de perte utilisée dans l’entraînement. Pour
l'évaluation des performances sur l'ensemble de test, outre la fonction de perte
vous vous intéresserez au PESQ et au STOI des voix estimées.

### Super-résolution

L'objectif de ce projet est d'augmenter la définition d'un signal audio
échantillonné à 4 kHz vers du 8kHz.

Une approche classique consisterait à ajouter un échantillon entre deux
échantillons du signal d'origine en interpolant les données (par une constante
ou un segment par exemple). On espère qu'un algorithme d'apprentissage pourrait
dépasser les performances de cette approche en intégrant de l'information plus
globale sur le signal. D'un point de vue spectral, le spectrogramme d'un signal
4 kHz est défini sur l'intervalle de fréquences [0, 2 kHz]. On peut transposer
ce spectrogramme sur l'intervalle [0, 4 kHz] qui correspond à un signal 8 kHz en
mettant toutes les fréquences entre 2 et 4 kHz à zéros. La tâche que l'on soumet
à l'algorithme est alors de deviner les fréquences [2 kHz, 4 kHz] à partir de la
partie basse du spectre. Il s'agit donc de compléter des données perdues.

Le dataset de train se compose de :

- Un dossier contenant des fichiers d'enregistrements de voix à 8kHz
  (audio/voice_origin/train) ; ce sont les données objectifs
- Un dossier contenant les fichiers correspondants à 4 kHz
  (audio/voice_4k/train) ; ce sont les données d'entrée à traiter

La correspondance entre les deux dossiers pour reconstituer les paires (donnée
d'entrée, vérité terrain) se fait par le nom de fichier. Les données de test
sont structurées de la même façon.

Vous pouvez au choix travailler sur spectrogramme ou directement sur la forme
d'onde en vous inspirant par exemple des références données pour la séparation
de sources (UNet, WavUnet)

Pour ce sujet vous implémenterez nécessairement une approche classique par
exemple en utilisant des interpolations et une méthode à base de réseaux de
neurones profonds.

Libre à vous de choisir la fonction de perte utilisée dans l’entraînement. Pour
l'évaluation des performances sur l'ensemble de test, outre la fonction de perte
vous vous intéresserez au PESQ et au STOI des voix estimées.

## Radio

Je vous propose deux sujets dans le prolongement de la classification de séries
temporelles par deep learning étudiée dans le TP3.

### Détection d'anomalies

L'objectif de ce projet est d'être capable dans la phase de test de détecter que
certains signaux n'ont pas été vus à l'entrainement. Dans la phase
d'entraînement vous partez de l'ensemble train.hdf5 du TP3 qui contient de
signaux

- labélisés 0, 1, 2, 3, 4, 5
- avec des SNR 30, 20, 10 ou 0 Vous avez en plus pour ce projet un dataset
  uniquement de test (radio/test_anomalies.hdf5)
- avec des signaux des classes 0 à 5 comme vues dans l'ensemble d'entrainement
  mais aussi avec des signaux de classes 6,7, et 8 qui ne sont PAS représentées
  dans l'ensemble d'entrainement.

L'objectif est de construire un algorithme entraîné sur les classes 0 à 5 et de
l'adapter pour qu'il soit capable en test de détecter si un signal a été vu ou
non à l'apprentissage. L'objectif en test est donc :

- de renvoyer 0 si le signal vient des classes 0 à 5
- de renvoyer 1 sinon Les performances seront évaluées en termes de précision et
  de recall (cf mini TP1 ). Toutes les classes de nouveautés peuvent être
  regroupées dans la même macro classe. Vous pourrez analyser vos résultats
  conditionnellement à la sous-classe de nouveauté mais il ne vous est pas
  demandé d'exploité cette information. La décision de cet algorithme pourra
  dépendre d'un seuil que vous pourrez faire varier.

Vous pouvez vous inspirer de ce qui est fait en détection de sons anormaux dans
le challenge DCASE :

- https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring
- https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

Vous pouvez aussi essayer d'appliquer des techniques de détection d'outliers ou
de nouveautés tels que ceux disponibles dans Scikit-learn
https://scikit-learn.org/stable/modules/outlier_detection.html dans un espace
latent obtenu par un réseau de neurones profonds.

Vous pouvez partir du ou des réseaux profonds que vous avez obtenu dans le TP3
ou le réentraîner avec une stratégie qui vous semble appropriée à la détection
d'anomalies.

### Few shot learning pour la classification de signaux

L'objectif de ce projet est de construire et entraîner un réseau de neurones à
identifier des classes avec peu de données. Vous pouvez partir sur le paradigme
enrôlement d'une classes et rattachement d'une donnée de test à une classe
enrôlée présenté dans le cours 09 et par exemple mis en oeuvre dans J. Snell et
Al., Prototypical Networks for Few-shot Learning, NIPS 2017 ou sur de
l'apprentissage de métrique ou toute autre méthode qui vous semble pertinente

Vouscdisposez des datasets suivants :

- L'ensemble train.hdf5 du TP3 qui contient des signaux de type 0, 1, 2, 3, 4, 5
  en grand nombre q
- Un ensemble radio/few_shots/enroll.hdf5 qui contient des (quelques) signaux de
  type 6, 7, 8, 9, 10, 11 ; ce sont les données que vous utiliserez pour
  "apprendre" ou "enrôler" les nouvelles classes
- Un ensemble radio/few_shots/test_fewshots.hdf5 qui contient des signaux de
  type 6, 7, 8, 9, 10, 11 ; vous testerez sur ces données vos performances de
  reconnaissance

Les données à disposition pour les nouvelles formes d'onde sont "généreuses"
pour que vous ne soyez pas limités mais vous êtes encouragés à n'utiliser si
possible qu'une partie de ces données ou de caractériser les performances de
votre approche en fonction du nombre de signaux que vous avez utilisés pour
"apprendre" les nouvelles classes.

En plus de la méthodologie d'entraînement vous pourrez vous inspirer de la
méthodologie d'analyse des performances de l'article de Snell et Al. pour
qualifier vos résultats. Vous pourrez notamment mettre en évidence le dépendance
des performances au nombre de données utilisées pour enrôlement les différentes
classes.
