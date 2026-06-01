public class TraitementSignal {

    // 1. Filtre Lisseur (Moyenneur) - Efface le bruit
    public static int[] appliquerFiltreMoyenneur(int[] pixelsBruts, int largeur, int hauteur) {
        int[] resultat = new int[pixelsBruts.length];
        for (int y = 1; y < hauteur - 1; y++) {
            for (int x = 1; x < largeur - 1; x++) {
                int somme = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int indexVoisin = (y + dy) * largeur + (x + dx);
                        somme += pixelsBruts[indexVoisin];
                    }
                }
                int indexActuel = y * largeur + x;
                resultat[indexActuel] = somme / 9;
            }
        }
        return resultat;
    }

    // 2. Filtre de Contours (Laplacien) - Garde uniquement les arêtes vives
    public static int[] appliquerFiltreContours(int[] pixelsBruts, int largeur, int hauteur) {
        int[] resultat = new int[pixelsBruts.length];
        int[][] noyau = {
                { -1, -1, -1 },
                { -1,  8, -1 },
                { -1, -1, -1 }
        };

        for (int y = 1; y < hauteur - 1; y++) {
            for (int x = 1; x < largeur - 1; x++) {
                int somme = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int indexVoisin = (y + dy) * largeur + (x + dx);
                        somme += pixelsBruts[indexVoisin] * noyau[dy + 1][dx + 1];
                    }
                }
                int indexActuel = y * largeur + x;
                resultat[indexActuel] = Math.max(0, Math.min(255, somme));
            }
        }
        return resultat;
    }

    // 3. Opération de Max Pooling (Réduction spatiale et tolérance)
    public static int[] appliquerMaxPooling(int[] pixelsBruts, int largeur, int hauteur, int tailleBloc) {
        int nouvelleLargeur = largeur / tailleBloc;
        int nouvelleHauteur = hauteur / tailleBloc;
        int[] resultat = new int[nouvelleLargeur * nouvelleHauteur];

        for (int y = 0; y < nouvelleHauteur; y++) {
            for (int x = 0; x < nouvelleLargeur; x++) {

                int valeurMax = 0;

                for (int dy = 0; dy < tailleBloc; dy++) {
                    for (int dx = 0; dx < tailleBloc; dx++) {
                        int pixelY = y * tailleBloc + dy;
                        int pixelX = x * tailleBloc + dx;

                        if (pixelY < hauteur && pixelX < largeur) {
                            int indexAncien = pixelY * largeur + pixelX;
                            int pixelActuel = pixelsBruts[indexAncien];
                            if (pixelActuel > valeurMax) {
                                valeurMax = pixelActuel;
                            }
                        }
                    }
                }
                int indexNouveau = y * nouvelleLargeur + x;
                resultat[indexNouveau] = valeurMax;
            }
        }
        return resultat;
    }

    // 4. Augmentation de données : Effet Miroir horizontal
    public static int[] appliquerMiroir(int[] pixelsBruts, int largeur, int hauteur) {
        int[] resultat = new int[pixelsBruts.length];

        for (int y = 0; y < hauteur; y++) {
            for (int x = 0; x < largeur; x++) {
                int indexOriginal = y * largeur + x;
                int indexMiroir = y * largeur + (largeur - 1 - x);
                resultat[indexMiroir] = pixelsBruts[indexOriginal];
            }
        }
        return resultat;
    }

    // 5. Outil d'ingénierie : Exporter un tableau de pixels en image JPG
    public static void exporterImage(int[] pixels, int largeur, int hauteur, String cheminSortie) {
        try {
            java.awt.image.BufferedImage img = new java.awt.image.BufferedImage(largeur, hauteur, java.awt.image.BufferedImage.TYPE_INT_RGB);

            for (int y = 0; y < hauteur; y++) {
                for (int x = 0; x < largeur; x++) {
                    int gris = pixels[y * largeur + x];
                    gris = Math.max(0, Math.min(255, gris));

                    int rgb = (gris << 16) | (gris << 8) | gris;
                    img.setRGB(x, y, rgb);
                }
            }

            javax.imageio.ImageIO.write(img, "jpg", new java.io.File(cheminSortie));
            System.out.println("📸 Preuve visuelle générée : " + cheminSortie);

        } catch (Exception e) {
            System.out.println("❌ Erreur lors de l'exportation de l'image : " + e.getMessage());
        }
    }
}