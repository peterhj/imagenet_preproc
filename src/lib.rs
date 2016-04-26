extern crate varraydb;

extern crate byteorder;
//extern crate chan;
extern crate csv;
extern crate image;
extern crate memmap;
extern crate tar;
extern crate threadpool;

extern crate rand;
extern crate rustc_serialize;
extern crate time;

use byteorder::{WriteBytesExt, LittleEndian};
use csv::{Reader as CsvReader};
use image::{GenericImage, ImageBuffer, Rgb, ColorType, ImageDecoder, DecodingResult};
use image::{DynamicImage, ImageFormat, load};
use image::imageops::{FilterType, resize};
//use image::jpeg::{JPEGDecoder};
use memmap::{Mmap, Protection};
use tar::{Archive};
use threadpool::{ThreadPool};
use varraydb::{VarrayDb};

use rand::{Rng, thread_rng};
use std::cmp::{min};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File};
use std::io::{Read, Seek, Write, Cursor};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread::{spawn};
use time::{get_time};

const NUM_ENCODERS: usize = 8;

#[derive(Clone, Debug)]
pub struct IlsvrcConfig {
  pub wnid_to_rawcats_path: PathBuf,
  pub train_archive_path:   PathBuf,
  pub valid_archive_path:   PathBuf,
  pub valid_rawcats_path:   PathBuf,

  pub resize_smaller_dim:   u32,
  pub keep_aspect_ratio:    bool,

  pub train_data_path:      PathBuf,
  pub train_labels_path:    PathBuf,
}

#[derive(RustcDecodable)]
struct WnidRawCatRecord {
  wnid: String,
  raw_cat: i32,
}

enum EncoderMsg {
  ImageFile(usize, i32, Vec<u8>),
}

enum WriterMsg {
  Skip(usize),
  ImageFile(usize, i32, Vec<u8>),
}

struct ReaderWorker {
  config:       IlsvrcConfig,
  encoder_txs:  Vec<SyncSender<EncoderMsg>>,
  wnid_to_rawcats:  HashMap<String, i32>,
  //archive_file:     File,
  //archive_map:      Mmap,
}

impl ReaderWorker {
  pub fn run(&mut self, archive_map: Mmap) {
    let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
    let mut train_archive = Archive::new(reader);
    let mut counter = 0;
    let start_time = get_time();
    for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
      let wnid_file = wnid_file.unwrap();
      let wnid_path: PathBuf = wnid_file.header().path().unwrap().into_owned();
      let mut wnid_archive = Archive::new(wnid_file);
      //println!("DEBUG: processing {} {:?}...", wnid_idx, wnid_path);
      for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
        let mut im_file = im_file.unwrap();

        let im_path = im_file.header().path().unwrap().into_owned();
        //let im_stem = im_path.file_stem().unwrap().to_str().unwrap();
        let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
        let im_stem_toks: Vec<_> = im_path_toks[0].splitn(2, "_").collect();
        if !self.wnid_to_rawcats.contains_key(im_stem_toks[0]) {
          println!("WARNING: unknown key: {} {} {:?} {:?}", wnid_idx, im_idx, im_path, im_stem_toks);
          continue;
        }
        let rawcat = *self.wnid_to_rawcats.get(im_stem_toks[0]).unwrap();

        let mut buf: Vec<u8> = vec![];
        match im_file.read_to_end(&mut buf) {
          Err(e) => panic!("failed to read image: {:?}", e),
          Ok(_) => {}
        }

        self.encoder_txs[counter % NUM_ENCODERS].send(
            EncoderMsg::ImageFile(counter, rawcat, buf),
        ).unwrap();
        counter += 1;
      }
      let lap_time = get_time();
      let elapsed = (lap_time - start_time).num_milliseconds() as f32 * 0.001;
      println!("DEBUG: processed: {} elapsed: {:.3}", counter, elapsed);
    }
  }
}

struct EncoderWorker {
  config:       IlsvrcConfig,
  encoder_rx:   Receiver<EncoderMsg>,
  writer_tx:    SyncSender<WriterMsg>,
}

impl EncoderWorker {
  pub fn run(&mut self) {
    loop {
      match self.encoder_rx.recv() {
        //Err(_) | Ok(EncoderMsg::Quit) => {
        Err(_) => {
          break;
        }
        Ok(EncoderMsg::ImageFile(idx, rawcat, buf)) => {
          let mut reader = Cursor::new(&buf);
          let mut image = None;
          match load(reader, ImageFormat::JPEG) {
            Err(_) => {}
            Ok(im) => image = Some(im),
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::PNG) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::GIF) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::TIFF) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            let mut reader = Cursor::new(&buf);
            match load(reader, ImageFormat::BMP) {
              Err(_) => {}
              Ok(im) => image = Some(im),
            }
          }
          if image.is_none() {
            println!("WARNING: decoding failed: {}", idx);
            self.writer_tx.send(
                WriterMsg::Skip(idx),
            ).unwrap();
            continue;
          }

          let image = image.unwrap();
          let image_dims = image.dimensions();
          let min_side = min(image_dims.0, image_dims.1);
          //println!("DEBUG: min side: {} dims: {:?} path: {:?}", min_side, image_dims, im_path);

          let mut encoded_image = vec![];
          if min_side <= self.config.resize_smaller_dim {
            let image_buf = image.to_rgb();
            let image = DynamicImage::ImageRgb8(image_buf);
            match image.save(&mut encoded_image, ImageFormat::PNG) {
              Err(e) => panic!("failed to encode image as png: {:?}", e),
              Ok(_) => {}
            }
          } else {
            let (old_width, old_height) = image_dims;
            let (new_width, new_height) = if old_width < old_height {
              (min_side, (min_side as f32 / old_width as f32 * old_height as f32).round() as u32)
            } else if old_width > old_height {
              ((min_side as f32 / old_height as f32 * old_width as f32).round() as u32, min_side)
            } else {
              (min_side, min_side)
            };

            let old_image_buf = image.to_rgb();
            let new_image_buf = resize(&old_image_buf, new_width as u32, new_height as u32, FilterType::Lanczos3);

            let new_image = DynamicImage::ImageRgb8(new_image_buf);
            match new_image.save(&mut encoded_image, ImageFormat::PNG) {
              Err(e) => panic!("failed to encode image as png: {:?}", e),
              Ok(_) => {}
            }
          }
          self.writer_tx.send(
              WriterMsg::ImageFile(idx, rawcat, encoded_image),
          ).unwrap();
        }
      }
    }
  }
}

struct WriterWorker {
  config:       IlsvrcConfig,
  writer_rx:    Receiver<WriterMsg>,
  counter:      usize,
}

impl WriterWorker {
  pub fn run(&mut self) {
    //let mut adj_idx = 0;
    let mut skip_set: HashSet<usize> = HashSet::new();
    let mut cache: BTreeMap<usize, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
    let mut data_db = VarrayDb::create(&self.config.train_data_path).unwrap();
    let mut labels_db = VarrayDb::create(&self.config.train_labels_path).unwrap();
    loop {
      match self.writer_rx.recv() {
        Err(_) => {
          break;
        }

        Ok(WriterMsg::Skip(idx)) => {
          skip_set.insert(idx);
          if idx == self.counter {
            //adj_idx += 1;
            self.counter += 1;
          }
          loop {
            if cache.contains_key(&self.counter) {
              let (buf, label_buf) = cache.remove(&self.counter).unwrap();
              data_db.append(&buf);
              labels_db.append(&label_buf);
              self.counter += 1;
            } else if skip_set.contains(&self.counter) {
              //adj_idx += 1;
              self.counter += 1;
            } else {
              break;
            }
          }
        }

        Ok(WriterMsg::ImageFile(idx, rawcat, buf)) => {
          //println!("DEBUG: images writen: {}", self.counter);
          let label_cat = rawcat - 1;
          assert!(label_cat >= 0 && label_cat < 1000);
          let mut label_buf = vec![];
          label_buf.write_i32::<LittleEndian>(label_cat).unwrap();
          assert_eq!(4, label_buf.len());

          if idx == self.counter {
            data_db.append(&buf);
            labels_db.append(&label_buf);
            self.counter += 1;
          } else {
            cache.insert(idx, (buf, label_buf));
            loop {
              if cache.contains_key(&self.counter) {
                let (buf, label_buf) = cache.remove(&self.counter).unwrap();
                data_db.append(&buf);
                labels_db.append(&label_buf);
                self.counter += 1;
              } else if skip_set.contains(&self.counter) {
                //adj_idx += 1;
                self.counter += 1;
              } else {
                break;
              }
            }
          }

          /*if self.counter % 100 == 0 {
            println!("DEBUG: images writen: {}", self.counter);
          }*/
        }
      }
    }
  }
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

  let archive_file = File::open(&config.train_archive_path).unwrap();
  let archive_map = Mmap::open(&archive_file, Protection::Read).unwrap();

  let mut samples_count = 0;
  {
    //let train_archive_file = File::open(&config.train_archive_path).unwrap();
    //let mut train_archive = Archive::new(train_archive_file);
    let mut reader = Cursor::new(unsafe { archive_map.as_slice() });
    let mut train_archive = Archive::new(reader);
    for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
      let wnid_file = wnid_file.unwrap();
      let wnid_path: PathBuf = wnid_file.header().path().unwrap().into_owned();
      //println!("DEBUG: counting {} {:?}...", wnid_idx, wnid_path);
      let mut wnid_archive = Archive::new(wnid_file);
      for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
        let im_file = im_file.unwrap();
        samples_count += 1;
      }
    }
  }
  println!("DEBUG: samples count: {}", samples_count);

  //let mut shuffle_idxs: Vec<_> = (0 .. samples_count).collect();
  //thread_rng().shuffle(&mut shuffle_idxs);

  let mut encoder_txs = vec![];
  let mut encoder_rxs = vec![];
  for _ in 0 .. NUM_ENCODERS {
    let (encoder_tx, encoder_rx) = sync_channel(128);
    encoder_txs.push(encoder_tx);
    encoder_rxs.push(Some(encoder_rx));
  }
  let (writer_tx, writer_rx) = sync_channel(128);

  let reader_thr = {
    let config = config.clone();
    spawn(move || {
      ReaderWorker{
        config:       config,
        encoder_txs:  encoder_txs,
        wnid_to_rawcats:  wnid_to_rawcats,
      }.run(archive_map);
    })
  };

  let encoder_barrier = Arc::new(Barrier::new(NUM_ENCODERS + 1));
  let encoder_pool = ThreadPool::new(NUM_ENCODERS);
  for i in 0 .. NUM_ENCODERS {
    let encoder_barrier = encoder_barrier.clone();
    let config = config.clone();
    let encoder_rx = encoder_rxs[i].take().unwrap();
    let writer_tx = writer_tx.clone();
    encoder_pool.execute(move || {
      EncoderWorker{
        config:     config,
        encoder_rx: encoder_rx,
        writer_tx:  writer_tx,
      }.run();
      encoder_barrier.wait();
    });
  }

  let writer_thr = {
    let config = config.clone();
    spawn(move || {
      WriterWorker{
        config:     config,
        writer_rx:  writer_rx,
        counter:    0,
      }.run();
    })
  };

  writer_thr.join();
  encoder_barrier.wait();
  reader_thr.join();

  /*let train_archive_file = File::open(&config.train_archive_path).unwrap();
  let mut train_archive = Archive::new(train_archive_file);
  for (wnid_idx, wnid_file) in train_archive.entries().unwrap().enumerate() {
    let wnid_file = wnid_file.unwrap();
    let wnid_path: PathBuf = wnid_file.header().path().unwrap().into_owned();
    let mut wnid_archive = Archive::new(wnid_file);
    println!("DEBUG: processing {} {:?}...", wnid_idx, wnid_path);
    for (im_idx, im_file) in wnid_archive.entries().unwrap().enumerate() {
      let mut im_file = im_file.unwrap();

      let im_path = im_file.header().path().unwrap().into_owned();
      //let im_stem = im_path.file_stem().unwrap().to_str().unwrap();
      let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
      let im_stem_toks: Vec<_> = im_path_toks[0].splitn(2, "_").collect();
      if !wnid_to_rawcats.contains_key(im_stem_toks[0]) {
        println!("WARNING: unknown key: {} {} {:?} {:?}", wnid_idx, im_idx, im_path, im_stem_toks);
        continue;
      }

      let mut buf: Vec<u8> = vec![];
      match im_file.read_to_end(&mut buf) {
        Err(e) => panic!("failed to read image: {:?}", e),
        Ok(_) => {}
      }
      let mut reader = Cursor::new(&buf);
      let mut image = None;
      match load(reader, ImageFormat::JPEG) {
        Err(_) => {}
        Ok(im) => image = Some(im),
      }
      if image.is_none() {
        let mut reader = Cursor::new(&buf);
        match load(reader, ImageFormat::PNG) {
          Err(_) => {}
          Ok(im) => image = Some(im),
        }
      }
      if image.is_none() {
        println!("WARNING: decoding failed: {:?}", im_path);
        continue;
      }

      let image = image.unwrap();
      let image_dims = image.dimensions();
      let min_dim = min(image_dims.0, image_dims.1);
      println!("DEBUG: min dim: {} dims: {:?} path: {:?}", min_dim, image_dims, im_path);
    }
  }*/
}
