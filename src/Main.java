import neurone.Neurone;
import neurone.NeuroneSigmoide;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import java.io.File;

public class Main {

    // Constantes de configuration
    private static final String MODELE_PATH = "modele_neurone.txt";
    private static final int TAILLE_BLOC = 4;
    private static final float MSE_LIMITE = 0.05f;
    public static final float ValeurNormalisation = 255.0f;
    private static Scanner scanner = new Scanner(System.in);

    // Options de prétraitement des images
    private static final boolean ACTIVER_NIVEAUX_DE_GRIS = true;
    private static final boolean ACTIVER_CONTOURS = true;
    private static final boolean ACTIVER_MAX_POOLING = true;
    private static final boolean ACTIVER_MELANGE_DONNE = true;
    private static final boolean ACTIVER_MIROIR = true;
    private static final boolean ACTIVER_NORMALISATION = true;

    public static void main(String[] args) {
        boolean quitter = false;
        while (!quitter) {
            System.out.println("\n--- MENU CLASSIFICATION CHATS VS CHIENS ---");
            System.out.println("1. Entraîner le modèle (Apprentissage + Augmentation)");
            System.out.println("2. Tester le modèle (Évaluation détaillée)");
            System.out.println("3. Quitter");
            System.out.print("Votre choix : ");

            String choix = scanner.nextLine();
            switch (choix) {
                case "1":
                    entrainerModele();
                    break;
                case "2":
                    testerModele();
                    break;
                case "3":
                    quitter = true;
                    System.out.println("Au revoir !");
                    break;
                default:
                    System.out.println("Choix invalide, veuillez réessayer.");
            }
        }
    }

    private static void entrainerModele() {
        System.out.println("\n--- ENTRAÎNEMENT DU MODÈLE ---");
        System.out.println("Chargement des images d'entraînement...");

        // Lecture des fichiers d'entraînement
        List<String> trainChats = Image.listeFichiers("dataset_animaux/train/cat/");
        List<String> trainChiens = Image.listeFichiers("dataset_animaux/train/dog/");

        if (trainChats.isEmpty() && trainChiens.isEmpty()) {
            System.out.println("Erreur : Aucun fichier trouvé dans dataset_animaux/train/");
            return;
        }

        // Attribution des labels : 0 pour les chats, 1 pour les chiens
        List<Image> imagesTrain = new ArrayList<>();
        for (String chemin : trainChats) imagesTrain.add(new Image(chemin, 0, ACTIVER_NIVEAUX_DE_GRIS));
        for (String chemin : trainChiens) imagesTrain.add(new Image(chemin, 1, ACTIVER_NIVEAUX_DE_GRIS));

        if (ACTIVER_MELANGE_DONNE){
            Collections.shuffle(imagesTrain);
        }

        // Calcul de la taille du set de données selon l'activation de l'augmentation
        int nbImagesOriginales = imagesTrain.size();
        int nbImagesTotales = ACTIVER_MIROIR ? (nbImagesOriginales * 2) : nbImagesOriginales;

        Image premiereImage = imagesTrain.get(0);
        int largeurInitiale = premiereImage.largeur();
        int hauteurInitiale = premiereImage.hauteur();

        int nbCanaux = ACTIVER_NIVEAUX_DE_GRIS ? 1 : 3;
        int blocEffectif = (ACTIVER_NIVEAUX_DE_GRIS && ACTIVER_MAX_POOLING) ? TAILLE_BLOC : 1;
        int tailleEntree = (largeurInitiale / blocEffectif) * (hauteurInitiale / blocEffectif) * nbCanaux;

        float[][] entreesTrain = new float[nbImagesTotales][tailleEntree];
        float[] resultatsTrain = new float[nbImagesTotales];

        float diviseur = ACTIVER_NORMALISATION ? ValeurNormalisation : 1.0f;

        if (ACTIVER_NIVEAUX_DE_GRIS) {
            System.out.println("Prétraitement en cours (" + nbImagesTotales + " images au total)...");
            imageTestOriginal(imagesTrain.get(0), largeurInitiale, hauteurInitiale);
        } else {
            System.out.println("Préparation des données RGB brutes (" + nbImagesTotales + " images)...");
        }

        // Constitution du dataset final avec application des filtres
        int idx = 0;
        for (int i = 0; i < nbImagesOriginales; i++) {
            Image img = imagesTrain.get(i);
            int label = img.label();

            if (ACTIVER_NIVEAUX_DE_GRIS) {
                resultatsTrain[idx] = label;
                int[] pixelsContours = ACTIVER_CONTOURS ?
                        TraitementSignal.appliquerFiltreContours(img.donnees(), largeurInitiale, hauteurInitiale) : img.donnees();
                int[] pixelsReduits = TraitementSignal.appliquerMaxPooling(pixelsContours, largeurInitiale, hauteurInitiale, blocEffectif);

                for (int j = 0; j < tailleEntree; j++) {
                    entreesTrain[idx][j] = pixelsReduits[j] / diviseur;
                }
                idx++;

                // Ajout de la version miroir si activée
                if (ACTIVER_MIROIR) {
                    resultatsTrain[idx] = label;
                    int[] pixelsMiroir = TraitementSignal.appliquerMiroir(img.donnees(), largeurInitiale, hauteurInitiale);
                    int[] pixelsContoursMiroir = ACTIVER_CONTOURS ?
                            TraitementSignal.appliquerFiltreContours(pixelsMiroir, largeurInitiale, hauteurInitiale) : pixelsMiroir;
                    int[] pixelsReduitsMiroir = TraitementSignal.appliquerMaxPooling(pixelsContoursMiroir, largeurInitiale, hauteurInitiale, blocEffectif);

                    for (int j = 0; j < tailleEntree; j++) {
                        entreesTrain[idx][j] = pixelsReduitsMiroir[j] / diviseur;
                    }
                    idx++;
                }
            } else {
                resultatsTrain[idx] = label;
                for (int j = 0; j < tailleEntree; j++) {
                    entreesTrain[idx][j] = img.donnees()[j] / diviseur;
                }
                idx++;

                if (ACTIVER_MIROIR) {
                    resultatsTrain[idx] = label;
                    for (int j = 0; j < tailleEntree; j++) {
                        entreesTrain[idx][j] = img.donnees()[j] / diviseur;
                    }
                    idx++;
                }
            }
        }

        // Phase d'apprentissage
        System.out.println("Début de l'apprentissage...");
        Neurone neurone = new NeuroneSigmoide(tailleEntree);
        neurone.apprentissage(entreesTrain, resultatsTrain, MSE_LIMITE);
        System.out.println("Apprentissage terminé.");

        System.out.println("Sauvegarde du modèle dans " + MODELE_PATH + "...");
        neurone.sauvegarde(MODELE_PATH);
    }

    private static void testerModele() {
        System.out.println("\n--- TEST DU MODÈLE ---");
        File file = new File(MODELE_PATH);
        if (!file.exists()) {
            System.out.println("Erreur : Aucun modèle trouvé. Veuillez d'abord entraîner le modèle (Option 1).");
            return;
        }

        // Chargement du dataset de test
        List<String> testChats = Image.listeFichiers("dataset_animaux/test/cat/");
        List<String> testChiens = Image.listeFichiers("dataset_animaux/test/dog/");

        if (testChats.isEmpty() && testChiens.isEmpty()) {
            System.out.println("Erreur : Aucun fichier trouvé dans dataset_animaux/test/");
            return;
        }

        List<Image> imagesTest = new ArrayList<>();
        for (String chemin : testChats) imagesTest.add(new Image(chemin, 0, ACTIVER_NIVEAUX_DE_GRIS));
        for (String chemin : testChiens) imagesTest.add(new Image(chemin, 1, ACTIVER_NIVEAUX_DE_GRIS));

        Image sample = imagesTest.get(0);
        int largeurInitiale = sample.largeur();
        int hauteurInitiale = sample.hauteur();

        int nbCanaux = ACTIVER_NIVEAUX_DE_GRIS ? 1 : 3;
        int blocEffectif = (ACTIVER_NIVEAUX_DE_GRIS && ACTIVER_MAX_POOLING) ? TAILLE_BLOC : 1;
        int tailleEntree = (largeurInitiale / blocEffectif) * (hauteurInitiale / blocEffectif) * nbCanaux;

        System.out.println("Chargement du modèle...");
        Neurone neurone = new NeuroneSigmoide(tailleEntree);
        neurone.chargement(MODELE_PATH);

        int bonnesReponses = 0;
        int totalTest = imagesTest.size();
        float diviseur = ACTIVER_NORMALISATION ? ValeurNormalisation : 1.0f;

        System.out.println("Évaluation détaillée sur " + totalTest + " images...");

        // Évaluation de chaque image
        for (int i = 0; i < totalTest; i++) {
            Image img = imagesTest.get(i);
            int labelReel = img.label();
            float[] entreeTest = new float[tailleEntree];

            // Application du même prétraitement que pour l'entraînement
            if (ACTIVER_NIVEAUX_DE_GRIS) {
                int[] pixelsContours = ACTIVER_CONTOURS ?
                        TraitementSignal.appliquerFiltreContours(img.donnees(), largeurInitiale, hauteurInitiale) : img.donnees();
                int[] pixelsReduits = TraitementSignal.appliquerMaxPooling(pixelsContours, largeurInitiale, hauteurInitiale, blocEffectif);

                for (int j = 0; j < tailleEntree; j++) {
                    entreeTest[j] = pixelsReduits[j] / diviseur;
                }
            } else {
                for (int j = 0; j < tailleEntree; j++) {
                    entreeTest[j] = img.donnees()[j] / diviseur;
                }
            }

            // Prédiction du réseau
            neurone.metAJour(entreeTest);
            float sortieBrute = neurone.sortie();
            int labelTrouve = (sortieBrute >= 0.5f) ? 1 : 0;

            String animalCible = (labelReel == 0) ? "Chat" : "Chien";
            String animalTrouve = (labelTrouve == 0) ? "Chat" : "Chien";

            if (labelReel == labelTrouve) {
                System.out.printf("✅ Image %d (%s) -> IA a trouvé : %s (Confiance: %.2f)\n", i, animalCible, animalTrouve, sortieBrute);
                bonnesReponses++;
            } else {
                System.out.printf("❌ Image %d (%s) -> IA s'est trompée : %s (Valeur sortie: %.2f)\n", i, animalCible, animalTrouve, sortieBrute);
            }
        }

        // Calcul et affichage du score de précision
        float pourcentage = ((float) bonnesReponses / totalTest) * 100;
        System.out.println("\n=========================================");
        System.out.println("Résultats sur " + totalTest + " images de test :");
        System.out.println("Score final : " + pourcentage + "% de bonnes réponses (" + bonnesReponses + "/" + totalTest + ")");
        System.out.println("=========================================");
    }

    private static int[] imageTestOriginal(Image img, int l, int h) {
        int[] pixelsTestOriginal = img.donnees();
        int[] pixelsTestMiroir = TraitementSignal.appliquerMiroir(pixelsTestOriginal, l, h);
        TraitementSignal.exporterImage(pixelsTestOriginal, l, h, "preuve_01_originale.jpg");
        TraitementSignal.exporterImage(pixelsTestMiroir, l, h, "preuve_02_miroir.jpg");
        return pixelsTestOriginal;
    }
}