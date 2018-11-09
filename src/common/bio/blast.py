from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML


def blast_seq(input_seq, only_first_match=False, alignments=500, descriptions=500, hitlist_size=50):
    """Returns BLAST results for given sequence as well as list of sequence titles

    Args:
      input_seq: protein sequence as string
      only_first_match: flag to return only first match (Default value = False)
      alignments: max number of aligments from BLAST (Default value = 500)
      descriptions: max number of descriptions to show (Default value = 500)
      hitlist_size: max number of hits to return. (Default value = 50)

    Returns:
      list of alignments as well as list of titles of sequences in the alignment results

    """
    seq = input_seq.replace('0', '')
    to_display, all_titles = [], []
    try:
        blast_record = get_blast_record(seq, alignments, descriptions, hitlist_size)
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                alignment_data = get_alignment_data(alignment, hsp)
                to_display.append(alignment_data)
                all_titles.append(alignment.title)
                if only_first_match:
                    break
    except Exception as e:
        print("Unexpected error when calling NCBIWWW.qblast:", str(e))
        to_display.append("Error!")
    return to_display, all_titles


def get_blast_record(seq, alignments, descriptions, hitlist_size):
    """Calls  NCBI's QBLAST server or a cloud service provider to get alignment results

    Args:
      alignments: max number of aligments from BLAST
      descriptions: max number of descriptions to show
      hitlist_size: max number of hits to return
      seq: protein sequence as string

    Returns:
      single Blast record

    """
    result_handle = NCBIWWW.qblast(program="blastp", database="nr", alignments=alignments,
                                   descriptions=descriptions,
                                   hitlist_size=hitlist_size, sequence=seq)
    blast_record = NCBIXML.read(result_handle)
    return blast_record


def get_alignment_data(alignment, hsp):
    """Formats aligment result

    Args:
      alignment: aligment info from BLAST
      hsp: HSP info

    Returns:
      formatted alignment output

    """
    return "****Alignment**** \nSequence: {} \nLength: {} | Score: {} | e value: {} | identities: {} \n{} \n{} \n{} \n".format(
        alignment.title, alignment.length, hsp.score, hsp.expect, hsp.identities, hsp.query, hsp.match, hsp.sbjct)
