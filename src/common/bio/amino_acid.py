from common.bio.constants import ID_TO_AMINO_ACID, AMINO_ACID_TO_ID, NON_STANDARD_AMINO_ACIDS


def from_amino_acid_to_id(data, column):
    """Converts sequences from amino acid to ids

    Args:
      data: data that contains amino acid that need to be converted to ids
      column: a column of the dataframe that contains amino acid that need to be converted to ids

    Returns:
      array of ids

    """
    return data[column].apply(lambda x: [AMINO_ACID_TO_ID[c] for c in x])


def from_id_from_amino_acid(data, column):
    """Converts sequences from ids to amino acid characters

    Args:
      data: data that contains ids that need to be converted to amino acid
      column: a column of the dataframe that contains ids that need to be converted to amino acid

    Returns:
      array of amino acid

    """
    return [[ID_TO_AMINO_ACID[id] for id in val] for index, val in data[column].iteritems()]


def filter_non_standard_amino_acids(data, column):
    """

    Args:
      data: dataframe containing amino acid sequence
      column: a column of dataframe that contains amino acid sequence

    Returns:
      filtered data drame

    """

    for amino_acid in NON_STANDARD_AMINO_ACIDS:
        data = data[~data[column].str.contains(amino_acid)]

    return data


def protein_seq_to_string(sequences, id_to_enzyme_class, labels=None, d_scores=None):
    """

    Args:
      sequences: Protein sequences
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)
      d_scores: Values of discriminator (Default value = None)

    Returns:
      array of strings with sequences and additional information

    """
    rows = []
    for index, seq in enumerate(sequences):
        sequence = "".join([ID_TO_AMINO_ACID[i] for i in seq])
        header = ""

        if labels is not None:
            header = "class: {}".format(id_to_enzyme_class[str(labels[index])])
        if d_scores is not None:
            header = "{} Discriminator score: {}".format(header, d_scores[index])
        rows.append("\> {} \n{}".format(header, sequence))

    return rows

def print_protein_seq(sequences, id_to_enzyme_class, labels=None, d_scores=None):
    """

    Args:
      sequences: Protein sequences
      id_to_enzyme_class: a dictionary to get enzyme class from its id
      labels: Ids  of Enzyme classes (Default value = None)
      d_scores: Values of discriminator (Default value = None)

    Returns:
      Signal for DONE

    """
    print("\n".join(protein_seq_to_string(sequences, id_to_enzyme_class, labels, d_scores)))
    return "DONE"
