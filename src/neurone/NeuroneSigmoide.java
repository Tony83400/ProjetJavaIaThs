package neurone;

public class NeuroneSigmoide extends Neurone
{
    // Fonction d'activation Sigmoïde (retourne une valeur douce entre 0.0 et 1.0)
    protected float activation(final float valeur) {
        return (float) (1.0 / (1.0 + Math.exp(-valeur)));
    }

    // Constructeur
    public NeuroneSigmoide(final int nbEntrees) {
        super(nbEntrees);
    }
}