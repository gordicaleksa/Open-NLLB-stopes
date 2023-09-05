"""
This scripts prepares 2 additional yaml configs "train_corpora.yaml", "valid_corpora.yaml" that will
contain filtered data files from the filtering pipeline stage.

The purpose is to make prepare_data.yaml config more readable and easier to maintain. Instead of dumping
this information directly into prepare_data.yaml config we instead link paths to yaml files.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import yaml


SUPPORTED_BCP_47_CODES = "ace_Arab,ace_Latn,acm_Arab,acq_Arab,aeb_Arab,afr_Latn,ajp_Arab,aka_Latn,amh_Ethi,apc_Arab,arb_Arab,ars_Arab,ary_Arab,arz_Arab,asm_Beng,ast_Latn,awa_Deva,ayr_Latn,azb_Arab,azj_Latn,bak_Cyrl,bam_Latn,ban_Latn,bel_Cyrl,bem_Latn,ben_Beng,bho_Deva,bjn_Arab,bjn_Latn,bod_Tibt,bos_Latn,bug_Latn,bul_Cyrl,cat_Latn,ceb_Latn,ces_Latn,cjk_Latn,ckb_Arab,crh_Latn,cym_Latn,dan_Latn,deu_Latn,dik_Latn,dyu_Latn,dzo_Tibt,ell_Grek,eng_Latn,epo_Latn,est_Latn,eus_Latn,ewe_Latn,fao_Latn,pes_Arab,fij_Latn,fin_Latn,fon_Latn,fra_Latn,fur_Latn,fuv_Latn,gla_Latn,gle_Latn,glg_Latn,grn_Latn,guj_Gujr,hat_Latn,hau_Latn,heb_Hebr,hin_Deva,hne_Deva,hrv_Latn,hun_Latn,hye_Armn,ibo_Latn,ilo_Latn,ind_Latn,isl_Latn,ita_Latn,jav_Latn,jpn_Jpan,kab_Latn,kac_Latn,kam_Latn,kan_Knda,kas_Arab,kas_Deva,kat_Geor,knc_Arab,knc_Latn,kaz_Cyrl,kbp_Latn,kea_Latn,khm_Khmr,kik_Latn,kin_Latn,kir_Cyrl,kmb_Latn,kon_Latn,kor_Hang,kmr_Latn,lao_Laoo,lvs_Latn,lij_Latn,lim_Latn,lin_Latn,lit_Latn,lmo_Latn,ltg_Latn,ltz_Latn,lua_Latn,lug_Latn,luo_Latn,lus_Latn,mag_Deva,mai_Deva,mal_Mlym,mar_Deva,min_Latn,mkd_Cyrl,plt_Latn,mlt_Latn,mni_Beng,khk_Cyrl,mos_Latn,mri_Latn,zsm_Latn,mya_Mymr,nld_Latn,nno_Latn,nob_Latn,npi_Deva,nso_Latn,nus_Latn,nya_Latn,oci_Latn,gaz_Latn,ory_Orya,pag_Latn,pan_Guru,pap_Latn,pol_Latn,por_Latn,prs_Arab,pbt_Arab,quy_Latn,ron_Latn,run_Latn,rus_Cyrl,sag_Latn,san_Deva,sat_Olck,scn_Latn,shn_Mymr,sin_Sinh,slk_Latn,slv_Latn,smo_Latn,sna_Latn,snd_Arab,som_Latn,sot_Latn,spa_Latn,als_Latn,srd_Latn,srp_Cyrl,ssw_Latn,sun_Latn,swe_Latn,swh_Latn,szl_Latn,tam_Taml,tat_Cyrl,tel_Telu,tgk_Cyrl,tgl_Latn,tha_Thai,tir_Ethi,taq_Latn,taq_Tfng,tpi_Latn,tsn_Latn,tso_Latn,tuk_Latn,tum_Latn,tur_Latn,twi_Latn,tzm_Tfng,uig_Arab,ukr_Cyrl,umb_Latn,urd_Arab,uzn_Latn,vec_Latn,vie_Latn,war_Latn,wol_Latn,xho_Latn,ydd_Hebr,yor_Latn,yue_Hant,zho_Hans,zho_Hant,zul_Latn"  # noqa
SUPPORTED_BCP_47_CODES = SUPPORTED_BCP_47_CODES.split(",")


prepare_data_configs_root_path = Path(__file__).parent / "prepare_data_configs"
os.makedirs(prepare_data_configs_root_path, exist_ok=True)


def prepare_configs(args):
    filtered_data_root_path = args.filtered_data_root_path
    train_primary_config_path = args.train_primary_config_path

    # Step 1: Create a mapping between lang direction & corpus name & src/trg to filtered data path
    lang_direction_to_filtered_data_path = defaultdict(dict)
    for root_dir, subdirs, files in os.walk(filtered_data_root_path):
        for file in files:
            lang_suffix = file.split(".")[-2]
            gz_suffix = file.split(".")[-1]
            if lang_suffix in SUPPORTED_BCP_47_CODES and gz_suffix == "gz":
                corpus_name = file.split(".")[0]
                direction = os.path.basename(root_dir)
                src, trg = direction.split("-")
                assert src in SUPPORTED_BCP_47_CODES, f"lang {src} not supported"
                assert trg in SUPPORTED_BCP_47_CODES, f"lang {trg} not supported"
                if src in file:
                    lang_direction_to_filtered_data_path[direction][f'{corpus_name}_{src}'] = os.path.join(root_dir, file)
                elif trg in file:
                    lang_direction_to_filtered_data_path[direction][f'{corpus_name}_{trg}'] = os.path.join(root_dir, file)
                else:
                    raise ValueError(f"file {file} does not contain neither {src} nor {trg}")

    # Step 2: Map the raw text data paths to filtered data paths in the train_primary.yaml config
    with open(train_primary_config_path, "r") as fout:
        config = yaml.safe_load(fout)
        for direction in config.keys():
            src, trg = direction.split("-")
            assert src in SUPPORTED_BCP_47_CODES, f"lang {src} not supported"
            assert trg in SUPPORTED_BCP_47_CODES, f"lang {trg} not supported"
            for corpus_name in config[direction].keys():
                dataset = config[direction][corpus_name]
                del dataset["fold"]
                del dataset["lang_dir"]
                del dataset["metadata"]
                del dataset["tsv"]
                dataset["src"] = lang_direction_to_filtered_data_path[direction][f'{corpus_name}_{src}']
                dataset["tgt"] = lang_direction_to_filtered_data_path[direction][f'{corpus_name}_{trg}']

        with open(prepare_data_configs_root_path / "train_corpora.yaml", "w") as fout:
            yaml.dump(config, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script to analyze certain statistics of the primary data."
    )
    parser.add_argument(
        "--filtered_data_root_path",
        "-d",
        type=str,
        required=True,
        help="Output directory of the filtering pipeline (where you stored your filtered data).",
    )
    parser.add_argument(
        "--train-primary-config-path",
        type=Path,
        default=Path("stopes/pipelines/filtering/filter_configs/unfiltered_corpora/train_primary.yaml"),
        help="Directory where the configuration files are stored.",
    )

    args = parser.parse_args()
    if not args.train_primary_config_path.exists():
        raise ValueError(f"File {args.train_primary_config_path} does not exist. Please run "
                         "the filtering stage (and `populate_data_conf.py` script) first.")
    prepare_configs(args)
