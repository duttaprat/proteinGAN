import re

import threading

from Bio import Entrez
from bio.amino_acid import protein_seq_to_string
from bio.blast import blast_seq
import tensorflow as tf


class BlastSummary(threading.Thread):
    def __init__(self, summary_writer, global_step, fake_seq, labels, id_to_enzyme_class, n_examples=2,
                 running_mode="train", d_scores=None):
        self._summary_writer = summary_writer
        self.fake_seq = fake_seq
        self.labels = labels
        self.global_step = global_step
        self.id_to_enzyme_class = id_to_enzyme_class
        self.n_examples = n_examples
        self.running_mode = running_mode
        self.d_scores = d_scores
        threading.Thread.__init__(self)

    def run(self):
        try:
            print("Running Blast thread")
            human_readable_fake = protein_seq_to_string(self.fake_seq, self.id_to_enzyme_class, self.labels,
                                                        self.d_scores)[:self.n_examples]
            text = self.get_blast_info(human_readable_fake)
            meta = tf.SummaryMetadata()
            meta.plugin_data.plugin_name = "text"
            summary = tf.Summary()
            summary.value.add(tag="BLAST_" + self.running_mode, metadata=meta,
                              tensor=tf.make_tensor_proto(text, dtype=tf.string))

            self._summary_writer.add_summary(summary, self.global_step)
            print("Finished Blasting for {} step".format(self.global_step))
        except Exception as e:
            print("Unexpected error in BlastSummary thread:", str(e))

    def get_blast_info(self, fake_seqs):
        info = []
        for index, fake_seq in enumerate(fake_seqs):
            to_display, all_titles = blast_seq(fake_seq.split("\n")[1], alignments=3, descriptions=3, hitlist_size=3)
            classes_string = self.get_enzyme_classes(all_titles)
            line = "{} \n \n{} \nEnzyme classes: \n{} \n ".format(fake_seq, "\r\n".join(to_display),
                                                                  classes_string)
            print(line.replace("\\", ""))
            info.append(line)
        return info

    def get_enzyme_classes(self, all_titles):
        matched = re.findall('gi\|(.+?)\|', "".join(all_titles))
        classes = []
        if len(matched) > 0:
            records = self.fetch_protein_data(matched)
            self.parse_enzyme_classes(classes, records)
        classes_string = ", ".join(classes) if len(classes) > 0 else "Not found"
        return classes_string

    def fetch_protein_data(self, matched):
        try:
            ids_to_search = ",".join(matched)
            handle = Entrez.efetch(db="Protein", id=ids_to_search, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print("Error when calling Entrez", str(e))
            records = []
        return records

    def parse_enzyme_classes(self, classes, records):
        for record in records:
            for e in record["GBSeq_feature-table"]:
                if e['GBFeature_key'] == "Protein":
                    for ee in e["GBFeature_quals"]:
                        if ee['GBQualifier_name'] == "EC_number":
                            classes.append(ee["GBQualifier_value"])
