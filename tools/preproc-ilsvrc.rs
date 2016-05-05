extern crate imagenet_preproc;

use imagenet_preproc::{IlsvrcConfig}; //, preproc_archive};

use std::path::{PathBuf};

fn main() {
  let config = IlsvrcConfig{
    wnid_to_rawcats_path: PathBuf::from("wnid_to_rawcats.csv"),
    train_archive_path:   PathBuf::from("/scratch/phj/data/ilsvrc2012_raw/ILSVRC2012_img_train.tar"),
    valid_archive_path:   PathBuf::from("/scratch/phj/data/ilsvrc2012_raw/ILSVRC2012_img_val.tar"),
    valid_rawcats_path:   PathBuf::from("ILSVRC2012_validation_ground_truth.txt"),

    resize_smaller_dim:   None, //Some(480),
    keep_aspect_ratio:    true,

    //train_data_path:      PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_orig_test/ilsvrc2012_scale480_train_data.varraydb"),
    //train_labels_path:    PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_orig_test/ilsvrc2012_scale480_train_labels.varraydb"),
    train_data_path:      PathBuf::from("/rscratch/phj/data/ilsvrc2012_noscalev2_orig/ilsvrc2012_maxscale480_orig_train_data.varraydb"),
    train_labels_path:    PathBuf::from("/rscratch/phj/data/ilsvrc2012_noscalev2_orig/ilsvrc2012_maxscale480_orig_train_labels.varraydb"),
    valid_data_path:      PathBuf::from("/rscratch/phj/data/ilsvrc2012_noscalev2_orig/ilsvrc2012_scale256_orig_valid_data.varraydb"),
    valid_labels_path:    PathBuf::from("/rscratch/phj/data/ilsvrc2012_noscalev2_orig/ilsvrc2012_scale256_orig_valid_labels.varraydb"),
  };

  config.preproc_train_data();
  //config.preproc_valid_data();
}
