# Classification d'Images Chat vs Chien avec un Unique Neurone

![Language](https://img.shields.io/badge/language-Java-orange.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

![Image](https://github.com/Tony83400/ProjetJavaIaThs/blob/main/readme_img.jpg)

Ce projet met en œuvre un système de classification d'images d'animaux à l'aide d'un seul neurone développés en Java. Il est conçu pour classer les images en deux catégories d'animaux : les chats et les chien.

## ✨ Fonctionnalités clés

*   **🧠 Intelligence Artificielle "From Scratch" :** Moteur d'apprentissage automatique construit sans bibliothèque externe, incluant l'algorithme de descente de gradient (MSE) et diverses fonctions d'activation (Sigmoïde, ReLU, Heavyside).
*   **🖼️ Traitement d'Image Avancé :** Extraction des caractéristiques via l'application de filtres de contours (Laplacien), de filtres moyenneurs, et réduction spatiale par Max Pooling.
*   **🔄 Augmentation de Données :** Génération virtuelle d'images supplémentaires par effet miroir horizontal pour renforcer l'entraînement du modèle.
*   **💾 Persistance du Modèle :** Sauvegarde automatique des poids synaptiques et du biais dans un fichier texte (modele_neurone.txt) pour réutilisation sans ré-entraînement.
*   **🖥️ Interface Interactive :** Menu en ligne de commande permettant de lancer l'entraînement, de générer des preuves visuelles du traitement, et d'évaluer précisément les performances du modèle.

## 📚 Technologies utilisées

*   **Langage:** Java
*   **Bibliothèques standards :** `java.io`, `java.nio.file` (pour la gestion des flux et des fichiers)
*   **Manipulation d'images :** `javax.imageio.ImageIO`, `java.awt.image.BufferedImage`
*   **Structure de données :** `java.util.List`, `java.util.stream.Collectors`

## Structure du projet
```
├── Main.java                 # Point d'entrée, orchestration et menu CLI interactif
├── Image.java                # Chargement, parsing RGB/Gris et métadonnées des images
├── TraitementSignal.java     # Boîte à outils mathématique : Filtres, Pooling, Export JPG
├── modele_neurone.txt        # Poids du réseau sauvegardés (généré post-entraînement)
├── dataset_animaux/          # Répertoire des datasets attendu par le programme
│   ├── train/                # Données d'entraînement (sous-dossiers cat/ et dog/)
│   └── test/                 # Données de validation (sous-dossiers cat/ et dog/)
└── neurone/                  # Package contenant la logique d'Intelligence Artificielle
    ├── iNeurone.java         # Contrat d'interface pour les neurones
    ├── Neurone.java          # Logique centrale (poids, biais, mise à jour, MSE)
    ├── NeuroneSigmoide.java  # Implémentation de l'activation Sigmoïde
    ├── NeuroneReLU.java      # Implémentation de l'activation ReLU
    └── NeuroneHeavyside.java # Implémentation de l'activation Heavyside
```

## 🚀 Prérequis et Installation

Pour obtenir une copie locale fonctionnelle, suivez ces étapes simples.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Tony83400/ProjetJavaIaThs.git
    cd ProjetJavaIaThs
    ```

2.  **Préparation du Dataset :** Le programme s'attend à trouver une arborescence de dossiers spécifique pour les images d'entraînement et de test. Créez la structure suivante à la racine du projet et placez-y vos images (`.jpg`, `.png`) :
    *   `dataset_animaux/train/cat/`
    *   `dataset_animaux/train/dog/`
    *   `dataset_animaux/test/cat/`
    *   `dataset_animaux/test/dog/`

3.  **Compilation :** Ouvrez un terminal à la racine du projet et compilez l'ensemble des fichiers Java :
    ```
    javac Main.java Image.java TraitementSignal.java neurone/*.java
    ```

## 💻 Configuration et Utilisation avec IntelliJ IDEA

Cette section vous guide pas à pas pour importer, configurer et exécuter ce projet de manière optimale en utilisant l'IDE **IntelliJ IDEA**.

### 1. Importation du projet
1. Lancez **IntelliJ IDEA**.
2. Dans la fenêtre d'accueil, cliquez sur **Open** (ou allez dans `File > Open...` si l'IDE est déjà ouvert).
3. Naviguez jusqu'au répertoire racine du projet `ProjetJavaIaThs` et cliquez sur **OK**.
4. L'IDE détectera automatiquement la configuration du projet grâce aux fichiers `.idea/` et au fichier `.iml` déjà présents.

### 2. Vérification du SDK Java (JDK)
Si le projet ne compile pas immédiatement, assurez-vous qu'un kit de développement Java (JDK) valide est associé au projet :
1. Allez dans `File > Project Structure...` (ou utilisez le raccourci `Ctrl+Alt+Shift+S` / `Cmd+;`).
2. Dans la section **Project**, vérifiez que le champ **SDK** pointe vers une version de Java (version 8 ou supérieure recommandée).
3. Si aucun SDK n'est configuré, cliquez sur le menu déroulant, faites `Add SDK > Download JDK...`, choisissez une version (ex: OpenJDK 17 ou 21) et cliquez sur **Download**.

### 3. Compilation et Build
* IntelliJ IDEA compile généralement le code automatiquement en arrière-plan.
* Pour forcer une recompilation complète de l'application, vous pouvez aller dans le menu du haut et sélectionner `Build > Build Project` (ou utiliser `Ctrl+F9` / `Cmd+F9`). Les fichiers compilés seront générés dans le dossier de sortie (généralement `out/production/`).

### 4. Exécution de l'application
1. Dans le panneau de gauche (*Project Tool Window*), développez l'arborescence pour localiser le fichier `Main.java` (situé à la racine des sources ou dans le dossier `src/` selon votre structure).
2. Double-cliquez sur `Main.java` pour l'ouvrir dans l'éditeur.
3. Pour lancer le programme, vous disposez de trois méthodes au choix :
   * Cliquez sur la petite **flèche verte** ▶️ située dans la marge de l'éditeur juste à gauche de la ligne `public static void main(String[] args)`.
   * Faites un **clic droit** n'importe où à l'intérieur du fichier de code `Main.java` et sélectionnez `Run 'Main.main()'`.
   * Utilisez le raccourci clavier `Ctrl+Shift+F10` (Windows/Linux) ou `Ctrl+Shift+R` (macOS) pendant que vous êtes sur le fichier.
4. Le menu de classification interactif (CLI) s'ouvrira alors directement dans l'onglet **Run** en bas de votre écran. Vous pourrez y saisir vos choix (`1`, `2` ou `3`) directement au clavier.

⚠️ **Note importante sur le Dataset :** Assurez-vous que le dossier `dataset_animaux/` est visible à la racine de l'arborescence d'IntelliJ. L'IDE utilise par défaut la racine du projet comme répertoire de travail, ce qui permettra au programme de trouver correctement les chemins relatifs vers vos images.

## ▶️ Utilisation
Une fois compilé, lancez le programme principal :
```
java Main
```
Le menu interactif s'affichera :
  * Tapez `1` pour démarrer le prétraitement, l'augmentation des données, l'entraînement du modèle Sigmoïde et sa sauvegarde. Des preuves visuelles des filtres (`preuve_01_originale.jpg` et `preuve_02_miroir.jpg`) seront générées à la racine.
  * Tapez `2` pour charger le modèle sauvegardé et tester sa précision sur les images du dossier de test. Un résumé statistique des prédictions s'affichera.
  * Tapez `3` pour quitter le programme.
