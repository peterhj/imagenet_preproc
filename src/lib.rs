extern crate csv;
extern crate image;
extern crate tar;

extern crate rand;
extern crate rustc_serialize;

use csv::{Reader as CsvReader};
use image::{GenericImage, ImageBuffer, Rgb, ColorType, ImageDecoder, DecodingResult};
use image::jpeg::{JPEGDecoder};
use tar::{Archive};

use rand::{Rng, thread_rng};
use std::collections::{HashMap};
use std::fs::{File};
use std::path::{Path, PathBuf};

pub struct IlsvrcConfig {
  pub wnid_to_rawcats_path: PathBuf,
  pub train_archive_path:   PathBuf,
  pub valid_archive_path:   PathBuf,
  pub valid_rawcats_path:   PathBuf,

  pub resize_smaller_dim:   u32,
  pub keep_aspect_ratio:    bool,
}

#[derive(RustcDecodable)]
struct WnidRawCatRecord {
  wnid: String,
  raw_cat: i32,
}

pub fn preproc_archive(config: &IlsvrcConfig) {
  let mut wnid_to_rawcats = HashMap::with_capacity(1000);
  {
    let mut reader = CsvReader::from_file(&config.wnid_to_rawcats_path).unwrap();
    for record in reader.decode() {
      let record: WnidRawCatRecord = record.unwrap();
      wnid_to_rawcats.insert(record.wnid, record.raw_cat);
    }
  }
  //println!("DEBUG: {:?}", wnid_to_rawcats);

  let mut samples_count = 0;
  {
    let train_archive_file = File::open(&config.train_archive_path).unwrap();
    let mut train_archive = Archive::new(train_archive_file);
    for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
      let wnid_file = wnid_file.unwrap();
      let mut wnid_archive = Archive::new(wnid_file);
      for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
        let im_file = im_file.unwrap();
        samples_count += 1;
      }
    }
  }
  println!("DEBUG: samples count: {}", samples_count);

  let mut shuffle_idxs: Vec<_> = (0 .. samples_count).collect();
  thread_rng().shuffle(&mut shuffle_idxs);

  let train_archive_file = File::open(&config.train_archive_path).unwrap();
  let mut train_archive = Archive::new(train_archive_file);
  for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
    let wnid_file = wnid_file.unwrap();
    let wnid_path: PathBuf = wnid_file.header().path().unwrap().into_owned();
    let mut wnid_archive = Archive::new(wnid_file);
    println!("DEBUG: processing {} {:?}...", wnid_idx, wnid_path);
    for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
      let im_file = im_file.unwrap();
      let im_path = im_file.header().path().unwrap().into_owned();
      //let im_stem = im_path.file_stem().unwrap().to_str().unwrap();
      let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
      let im_stem_toks: Vec<_> = im_path_toks[0].splitn(2, "_").collect();
      if !wnid_to_rawcats.contains_key(im_stem_toks[0]) {
        println!("WARNING: unknown key: {} {} {:?} {:?}", wnid_idx, im_idx, im_path, im_stem_toks);
        continue;
      }
      let mut decoder = JPEGDecoder::new(im_file);
      match decoder.colortype() {
        Ok(ColorType::RGB(8)) => {}
        Ok(x) => {
          println!("WARNING: decoding unsupported colortype: {:?} {:?}", x, im_path);
          continue;
        }
        Err(e) => {
          println!("WARNING: decoding colortype error: {:?} {:?}", e, im_path);
          continue;
        }
      };
      let (w, h) = match decoder.dimensions() {
        Ok((w, h)) => (w, h),
        Err(e) => {
          println!("WARNING: decoding dims error: {:?} {:?}", e, im_path);
          continue;
        }
      };
      let pixels = match decoder.read_image() {
        Ok(DecodingResult::U8(pixels)) => pixels,
        Ok(DecodingResult::U16(_)) => {
          println!("WARNING: decoding unsupported u16 pixels: {:?}", im_path);
          continue;
        }
        Err(e) => {
          println!("WARNING: decoding error: {:?} {:?}", e, im_path);
          continue;
        }
      };
      let im_buf = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(w, h, pixels).unwrap();
    }
  }
}
