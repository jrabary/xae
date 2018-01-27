from nets import dcgan


def encode(x, ):
    dcgan.generator()

def decode(z, observable_space_dim, is_training=True):
    """Use dcgan generator architecture as decoder"""
    logits, _ = dcgan.generator(z, final_size=observable_space_dim, is_training=is_training)
    return logits
