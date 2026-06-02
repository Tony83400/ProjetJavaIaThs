import neurone.Neurone;
import neurone.NeuroneSigmoide;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import java.io.File;

public class Main {

    private static final String MODELE_PATH = "modele_neurone.txt";
    private static final int TAILLE_BLOC = 4;
    private static final float MSE_LIMITE = 0.05f;
    private static Scanner scanner = new Scanner(System.in);

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

        List<String> trainChats = Image.listeFichiers("dataset_animaux/train/cat/");
        List<String> trainChiens = Image.listeFichiers("dataset_animaux/train/dog/");

        if (trainChats.isEmpty() && trainChiens.isEmpty()) {
            System.out.println("Erreur : Aucun fichier trouvé dans dataset_animaux/train/");
            return;
        }

        List<Image> imagesTrain = new ArrayList<>();
        for (String chemin : trainChats) imagesTrain.add(new Image(chemin, 0, true));
        for (String chemin : trainChiens) imagesTrain.add(new Image(chemin, 1, true));

        Collections.shuffle(imagesTrain);

        int nbImagesOriginales = imagesTrain.size();
        int nbImagesTotales = nbImagesOriginales * 2;

        Image premiereImage = imagesTrain.get(0);
        int largeurInitiale = premiereImage.largeur();
        int hauteurInitiale = premiereImage.hauteur();
        int tailleEntree = (largeurInitiale / TAILLE_BLOC) * (hauteurInitiale / TAILLE_BLOC);

        float[][] entreesTrain = new float[nbImagesTotales][tailleEntree];
        float[] resultatsTrain = new float[nbImagesTotales];

        System.out.println("Prétraitement et Augmentation (Miroir) en cours (" + nbImagesTotales + " images virtuelles)...");

        // --- GÉNÉRATION DE LA PREUVE VISUELLE SUR LA 1ÈRE IMAGE ---
        Image imageTest = imagesTrain.get(0);
        int[] pixelsTestOriginal = imageTest.donnees();
        int[] pixelsTestMiroir = TraitementSignal.appliquerMiroir(pixelsTestOriginal, largeurInitiale, hauteurInitiale);

        TraitementSignal.exporterImage(pixelsTestOriginal, largeurInitiale, hauteurInitiale, "preuve_01_originale.jpg");
        TraitementSignal.exporterImage(pixelsTestMiroir, largeurInitiale, hauteurInitiale, "preuve_02_miroir.jpg");
        // -----------------------------------------------------------

        for (int i = 0; i < nbImagesOriginales; i++) {
            Image img = imagesTrain.get(i);
            int label = img.label();

            // --- 1. TRAITEMENT DE L'IMAGE ORIGINALE ---
            resultatsTrain[2 * i] = label;
            int[] pixelsContours = TraitementSignal.appliquerFiltreContours(img.donnees(), largeurInitiale, hauteurInitiale);
            int[] pixelsReduits = TraitementSignal.appliquerMaxPooling(pixelsContours, largeurInitiale, hauteurInitiale, TAILLE_BLOC);

            for (int j = 0; j < tailleEntree; j++) {
                entreesTrain[2 * i][j] = pixelsReduits[j] / 255.0f;
            }

            // --- 2. TRAITEMENT DE L'IMAGE MIROIR ---
            resultatsTrain[2 * i + 1] = label;
            int[] pixelsMiroir = TraitementSignal.appliquerMiroir(img.donnees(), largeurInitiale, hauteurInitiale);
            int[] pixelsContoursMiroir = TraitementSignal.appliquerFiltreContours(pixelsMiroir, largeurInitiale, hauteurInitiale);
            int[] pixelsReduitsMiroir = TraitementSignal.appliquerMaxPooling(pixelsContoursMiroir, largeurInitiale, hauteurInitiale, TAILLE_BLOC);

            for (int j = 0; j < tailleEntree; j++) {
                entreesTrain[2 * i + 1][j] = pixelsReduitsMiroir[j] / 255.0f;
            }
        }

        System.out.println("Début de l'apprentissage (cela peut prendre un peu plus de temps)...");
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

        List<String> testChats = Image.listeFichiers("dataset_animaux/test/cat/");
        List<String> testChiens = Image.listeFichiers("dataset_animaux/test/dog/");

        if (testChats.isEmpty() && testChiens.isEmpty()) {
            System.out.println("Erreur : Aucun fichier trouvé dans dataset_animaux/test/");
            return;
        }

        List<Image> imagesTest = new ArrayList<>();
        for (String chemin : testChats) imagesTest.add(new Image(chemin, 0, true));
        for (String chemin : testChiens) imagesTest.add(new Image(chemin, 1, true));

        Image sample = imagesTest.get(0);
        int largeurInitiale = sample.largeur();
        int hauteurInitiale = sample.hauteur();
        int tailleEntree = (largeurInitiale / TAILLE_BLOC) * (hauteurInitiale / TAILLE_BLOC);

        System.out.println("Chargement du modèle...");
        Neurone neurone = new NeuroneSigmoide(tailleEntree);
        neurone.chargement(MODELE_PATH);

        int bonnesReponses = 0;
        int totalTest = imagesTest.size();

        System.out.println("Évaluation détaillée sur " + totalTest + " images...");
        for (int i = 0; i < totalTest; i++) {
            Image img = imagesTest.get(i);
            int labelReel = img.label();

            float[] entreeTest = new float[tailleEntree];
            int[] pixelsContours = TraitementSignal.appliquerFiltreContours(img.donnees(), largeurInitiale, hauteurInitiale);
            int[] pixelsReduits = TraitementSignal.appliquerMaxPooling(pixelsContours, largeurInitiale, hauteurInitiale, TAILLE_BLOC);

            for (int j = 0; j < tailleEntree; j++) {
                entreeTest[j] = pixelsReduits[j] / 255.0f;
            }

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

        float pourcentage = ((float) bonnesReponses / totalTest) * 100;
        System.out.println("\n=========================================");
        System.out.println("Résultats sur " + totalTest + " images de test :");
        System.out.println("Score final : " + pourcentage + "% de bonnes réponses (" + bonnesReponses + "/" + totalTest + ")");
        System.out.println("=========================================");
    }
}