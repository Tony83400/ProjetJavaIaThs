import neurone.iNeurone;
import neurone.NeuroneSigmoide;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

public class TestClassification {

    final static float MSElimite = 0.05f;

    public static void main(String[] args) {

        // --- PARTIE 1 : PRÉPARATION DES DONNÉES D'ENTRAÎNEMENT (TRAIN) ---
        System.out.println("1. Chargement de TOUTES les images d'entraînement (Train)...");

        List<String> trainChats = Image.listeFichiers("dataset_animaux/train/cat/");
        List<String> trainChiens = Image.listeFichiers("dataset_animaux/train/dog/");

        List<Image> imagesTrain = new ArrayList<>();
        System.out.println("-> Chargement de " + trainChats.size() + " chats...");
        for (String chemin : trainChats) imagesTrain.add(new Image(chemin, 0, true));

        System.out.println("-> Chargement de " + trainChiens.size() + " chiens...");
        for (String chemin : trainChiens) imagesTrain.add(new Image(chemin, 1, true));

        // Mélange indispensable pour un bon apprentissage
        Collections.shuffle(imagesTrain);

        int nbImagesTrain = imagesTrain.size();
        System.out.println("Total des images d'entraînement : " + nbImagesTrain);

        // --- GESTION DES DIMENSIONS APRES POOLING ---
        int largeurInitiale = imagesTrain.get(0).largeur();
        int hauteurInitiale = imagesTrain.get(0).hauteur();

        int tailleBloc = 4;
        int nouvelleLargeur = largeurInitiale / tailleBloc;
        int nouvelleHauteur = hauteurInitiale / tailleBloc;
        int tailleEntreeNeurone = nouvelleLargeur * nouvelleHauteur;

        float[][] entreesTrain = new float[nbImagesTrain][tailleEntreeNeurone];
        float[] resultatsTrain = new float[nbImagesTrain];

        System.out.println("-> Traitement du signal en cours (Contours + Pooling)...");
        for (int i = 0; i < nbImagesTrain; i++) {
            Image img = imagesTrain.get(i);
            resultatsTrain[i] = img.label();

            int[] pixelsContours = TraitementSignal.appliquerFiltreContours(img.donnees(), largeurInitiale, hauteurInitiale);
            int[] pixelsReduits = TraitementSignal.appliquerMaxPooling(pixelsContours, largeurInitiale, hauteurInitiale, tailleBloc);

            for (int j = 0; j < tailleEntreeNeurone; j++) {
                entreesTrain[i][j] = pixelsReduits[j] / 255.0f;
            }
        }

        // --- PARTIE 2 : APPRENTISSAGE ---
        System.out.println("\n2. Début de l'apprentissage sur " + nbImagesTrain + " images (Patientez)...");
        final iNeurone neurone = new NeuroneSigmoide(tailleEntreeNeurone);

        neurone.apprentissage(entreesTrain, resultatsTrain, MSElimite);
        System.out.println("Apprentissage terminé.");

        // --- PARTIE 3 : L'ÉPREUVE DE VÉRITÉ (TEST) ---
        System.out.println("\n3. Chargement de TOUTES les images de TEST (Jamais vues par l'IA)...");
        List<String> testChats = Image.listeFichiers("dataset_animaux/test/cat/");
        List<String> testChiens = Image.listeFichiers("dataset_animaux/test/dog/");

        List<Image> imagesTest = new ArrayList<>();
        for (String chemin : testChats) imagesTest.add(new Image(chemin, 0, true));
        for (String chemin : testChiens) imagesTest.add(new Image(chemin, 1, true));

        int bonnesReponses = 0;
        int totalTest = imagesTest.size();

        System.out.println("\n--- ÉVALUATION DÉTAILLÉE EN COURS ---");
        for (int i = 0; i < totalTest; i++) {
            Image img = imagesTest.get(i);
            int labelReel = img.label();

            float[] entreeTest = new float[tailleEntreeNeurone];
            int[] pixelsContours = TraitementSignal.appliquerFiltreContours(img.donnees(), largeurInitiale, hauteurInitiale);
            int[] pixelsReduits = TraitementSignal.appliquerMaxPooling(pixelsContours, largeurInitiale, hauteurInitiale, tailleBloc);

            for (int j = 0; j < tailleEntreeNeurone; j++) {
                entreeTest[j] = pixelsReduits[j] / 255.0f;
            }

            neurone.metAJour(entreeTest);
            float sortieBrute = neurone.sortie();
            int labelTrouve = (sortieBrute >= 0.5f) ? 1 : 0;

            String animalCible = (labelReel == 0) ? "Chat" : "Chien";
            String animalTrouve = (labelTrouve == 0) ? "Chat" : "Chien";

            if (labelReel == labelTrouve) {
                // Succès
                System.out.printf("✅ Image %d (%s) -> IA a trouvé : %s (Confiance: %.2f)\n", i, animalCible, animalTrouve, sortieBrute);
                bonnesReponses++;
            } else {
                // Échec
                System.out.printf("❌ Image %d (%s) -> IA s'est trompée : %s (Valeur sortie: %.2f)\n", i, animalCible, animalTrouve, sortieBrute);
            }
        }

        // --- PARTIE 4 : SCORE FINAL ---
        float pourcentage = ((float) bonnesReponses / totalTest) * 100;
        System.out.println("\n=========================================");
        System.out.println("Résultats sur " + totalTest + " images de test :");
        System.out.println("Score final de l'IA : " + pourcentage + "% de bonnes réponses.");
        System.out.println("=========================================");
    }
}