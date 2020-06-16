"""
from data_processing import create_act_inact_files_for_all_targets
from data_processing import create_act_inact_files_similarity_based_neg_enrichment_threshold
from data_processing import create_preprocessed_bioact_file
from data_processing import save_comp_imgs_from_smiles
from data_processing import create_final_randomized_training_val_test_sets
"""
from data_processing import create_preprocessed_bioact_file
from data_processing import get_chembl_uniprot_sp_id_mapping
from data_processing import get_chemblid_smiles_inchi_dict
from data_processing import DEEPScreenDataset, get_train_test_val_data_loaders

# source activate new-rdkit-env
# STEP 1
# Generate filtered dataset

# mysql> create database chembl_27 DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;
# mysql --user=root --password=12345 chembl_27 < chembl_27_mysql.dmp
# ./mysql --user=root --password=12345 --database=chembl_27 </Users/trman/OneDrive\ -\ ceng.metu.edu.tr/Projects/BioactivityDataAnalysis/bin/SQLQueries/chembl27_sp_b_pchembl.sql> /Users/trman/OneDrive\ -\ ceng.metu.edu.tr/Projects/BioactivityDataAnalysis/trainingFiles/ChEMBL/chembl27_raw_filtered_sp_b_pchembl_data.txt

# STEP 2
# remove dublicates and create preprossed dataset
# create_preprocessed_bioact_file("chembl27_raw_filtered_sp_b_pchembl_data.txt", "chembl27")
########### DEEPScreen Training Steps ###########

# STEP 3 Create act inact files without negative enrichment
# create_act_inact_files_for_all_targets("chembl27_preprocessed_filtered_bioactivity_dataset.tsv", "chembl27", 10.0, 20.0)

# STEP 4 create chembl_id uniprot mapping file and download fasta sequences from UniProt
# python test_data_processing.py > ../new_training_files/chembl27_sp_chemblid_uniprotisetd_mapping.tsv
"""
chembl_uniprot_sp_id_mapping_dict = get_chembl_uniprot_sp_id_mapping("chembl27_uniprot_mapping.txt")
for chembl_id in chembl_uniprot_sp_id_mapping_dict:
    print("{}\t{}".format(chembl_id, chembl_uniprot_sp_id_mapping_dict[chembl_id][0]))
"""
# STEP 5 get uniprot accs from chembl27_sp_chemblid_uniprotid_mapping and download protein sequences from uniprot


# STEP 6 create chembl sequences against chembl sequences blast output file
"""
/Users/trman/OneDrive\ -\ ceng.metu.edu.tr/Projects/UniGOPred/makeblastdb -dbtype prot -in ../new_training_files/chembl27_sp_targets_uniprot_sequences.fasta -out ../new_training_files/chembl27_sp_targets_uniprot_sequences.fasta.blastdb


/Users/trman/OneDrive\ -\ ceng.metu.edu.tr/Projects/UniGOPred/blastp -query ../new_training_files/chembl27_sp_targets_uniprot_sequences.fasta -db  ../new_training_files/chembl27_sp_targets_uniprot_sequences.fasta.blastdb  -outfmt 6 -out ../new_training_files/chembl27_uniprot_sequences_against_chembl27_uniprot_sequences_blast.out
"""

# STEP 7 create final negative enriched active inactive file
# create_act_inact_files_similarity_based_neg_enrichment_threshold("chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0.tsv", "chembl27_uniprot_sequences_against_chembl27_uniprot_sequences_blast.out", 0.20)

# STEP 8 Crete train test validation split and images
# # create_final_randomized_training_val_test_sets("chembl27_preprocessed_filtered_act_inact_comps_10.0_20.0_blast_comp_0.2.txt", "chembl27_chemreps.txt")
# get_chemblid_smiles_inchi_dict("chembl27_chemreps.txt")

#  save_comp_imgs_from_smiles("test", "CHEMBL153534", "Cc1cc(-c2csc(N=C(N)N)n2)cn1C")

# my_dataset = DEEPScreenDataset("CHEMBL202", "training")
# my_dataset = DEEPScreenDataset("CHEMBL202", "validation")
# my_dataset = DEEPScreenDataset("CHEMBL202", "test")
# print(len(my_dataset))
# print(my_dataset.__getitem__(1))
# get_train_test_val_data_loaders("CHEMBL202")
"""
import cv2

img_arr = cv2.imread(
    "/Users/trman/OneDrive - ceng.metu.edu.tr/Projects/DEEPScreen/new_training_files/target_training_datasets/CHEMBL202/imgs/CHEMBL36.png")
cv2.imshow("asdasd", img_arr)
print(img_arr.shape)
cv2.waitKey(0)
angle = 45
rows, cols, channel = img_arr.shape
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
rotated_image_array = cv2.warpAffine(img_arr, rotation_matrix, (cols, rows), cv2.INTER_LINEAR,
                                     borderValue=(255, 255, 255))  # cv2.BORDER_CONSTANT, 255)
cv2.imshow("asdasd", rotated_image_array)
cv2.waitKey(0)
"""
