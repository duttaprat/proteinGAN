from gan.image.image import Image
from gan.protein.blast_hook import BlastHook
from gan.protein.protein import Protein
from gan.sngan.model import SNGAN
from gan.wgan.model import WGAN


def get_model(flags, properties):
    if flags.model_type == "sngan":
        if "image" in flags.dataset:
            gan = SNGAN(Image(flags, properties))
        elif "protein" in flags.dataset:
            gan = SNGAN(Protein(flags, properties))
        else:
            raise NotImplementedError

    elif flags.model_type == "wgan":
        if flags.dataset == "mnist":
            gan = WGAN(Image(flags, properties))
        elif "protein" in flags.dataset:
            gan = WGAN(Protein(flags, properties))
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    return gan


def get_specific_hooks(flags, logdir, properties):
    hooks = []
    if "protein" in flags.dataset:
        # print("No Blast hook")
        id_to_enzyme_class_dict = properties["class_mapping"]
        hooks.append(
            BlastHook(id_to_enzyme_class_dict, every_n_steps=flags.steps_for_blast,
                      output_dir=logdir, n_examples=2))
        # hooks.append(VariableRestorer("../../logs/tcn_sequence/v1", ACID_EMBEDDINGS, "embedding/acid_embeddings"))

    return hooks
