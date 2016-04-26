extern crate imagenet_preproc;

use imagenet_preproc::{IlsvrcConfig, preproc_archive};

use std::path::{PathBuf};

fn main() {
  let config = IlsvrcConfig{
    wnid_to_rawcats_path: PathBuf::from("wnid_to_rawcats.csv"),
    train_archive_path:   PathBuf::from("/scratch/phj/data/ilsvrc2012_raw/ILSVRC2012_img_train.tar"),
    valid_archive_path:   PathBuf::from("/scratch/phj/data/ilsvrc2012_raw/ILSVRC2012_img_val.tar"),
    valid_rawcats_path:   PathBuf::from("ILSVRC2012_validation_ground_truth.txt"),

    resize_smaller_dim:   480,
    keep_aspect_ratio:    true,

    train_data_path:      PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480/ilsvrc2012_scale480_train_data.varraydb"),
    train_labels_path:    PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480/ilsvrc2012_scale480_train_labels.varraydb"),
  };

  preproc_archive(&config);
}
