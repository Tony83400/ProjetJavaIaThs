package neurone;

public class NeuroneReLU extends Neurone
{
    // Fonction d'activation ReLU (retourne 0 si négatif, ou la valeur exacte si positif)
    protected float activation(final float valeur) {
        return Math.max(0.0f, valeur);
    }

    // Constructeur
    public NeuroneReLU(final int nbEntrees) {
        super(nbEntrees);
    }
}